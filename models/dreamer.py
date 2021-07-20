import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.nn import Module, Linear
from torch.nn import LeakyReLU, ReLU, Conv2d, ConvTranspose2d, Tanh
from torch.nn import BatchNorm2d, Dropout, Dropout2d, Flatten
from torch.distributions import Categorical

from collections import namedtuple
from itertools import count
from copy import deepcopy

from .autoencoders import CnnAE
from .datasets import MyDatasetSeq


means = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
stds = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
to_tensor = torchvision.transforms.Compose([
                                #torchvision.transforms.Resize(50, 50),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                                        mean=means,
                                                        std=stds)
                             ])
def new_state_encoded(z, action_model, action, num_classes):
    action_onehot = F.one_hot(action, num_classes=num_classes).type(torch.float)

    cond_action = torch.cat([z, action_onehot], 1)
    encoded_action = action_model(cond_action).type(torch.float)
    return z + encoded_action

resize_ = 40
class Dreamer(torch.nn.Module):
    def __init__(self, encoder, decoder, action_space_size, latent_size=11,
                 gamma=0.99, horizon=12, L=15, replay_buffer=None, lambda_=0.95):
        super().__init__()
        self.gamma = gamma
        self.latent_size = latent_size
        self.action_space_size = action_space_size
        self.encoder = encoder
        self.decoder = decoder
        self.ae = CnnAE(encoder, decoder)
        self.reward_model = self.build_mlp(input_size=self.latent_size)
        self.transition_model = self.build_mlp(input_size=self.latent_size + action_space_size, 
                                               output_size=latent_size)
        self.world_model_optim = self.build_world_model_optim()
        self.latent_size = encoder.latent_size
        self.H = horizon
        self.L = L
        self.lambda_ = lambda_
        self.replay_buffer = replay_buffer

        self.actor = self.build_mlp(input_size=self.latent_size, output_size=action_space_size)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic = self.build_mlp(input_size=self.latent_size)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.mse_loss = nn.MSELoss()

    def build_world_model_optim(self):
        world_model_params = [x for x in self.encoder.parameters()] + [x for x in self.decoder.parameters()]
        world_model_params = world_model_params + [x for x in self.reward_model.parameters()]
        world_model_params = world_model_params + [x for x in self.transition_model.parameters()]
        return optim.Adam(world_model_params, lr=1e-3)
    
    def build_mlp(self, input_size, output_size=1):
        layers = []
        layers.append(nn.Linear(input_size, self.latent_size * 2))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.latent_size*2, 10))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(10, output_size))

        return nn.Sequential(*layers)

    def train_world_model(self, close_embedding_beta=0, diff_b=0.1):
        mse = nn.MSELoss()
        loader = self.replay_buffer
        world_model_optim = self.world_model_optim

        for batch_idx, (state, action, reward, next_state) in enumerate(loader):

            world_model_optim.zero_grad()
            
            for t in range(self.L):
                state_t = state[:, t,:, :, :]
                action_t = action[:, t]
                reward_t = reward[:, t].float()
                next_state_t = next_state[:, t, :, :, :]
                
                z, mu, logvar = self.encoder(state_t)
                
                encoded_next_state_hat = new_state_encoded(z, self.transition_model,
                                                          action_t, self.action_space_size)
                next_state_hat = self.decoder(encoded_next_state_hat)
                recon_loss = mse(next_state_hat, next_state_t)


                reward_hat = self.reward_model(z)
                reward_loss = mse(reward_t, reward_hat)

                # remains to implement kl loss
                # kl_loss = torch.distributions.kl.kl_divergence(z, next_state_hat)
                enc_next_state, mu, logvar = self.encoder(next_state_t)
                #enc_next_state = torch.flatten(enc_next_state)
 
                encoded_next_state_hat = torch.reshape(encoded_next_state_hat,
                                                       enc_next_state.shape)
                print("enc_ {}".format(enc_next_state.shape))
                print("encoded {}".format(encoded_next_state_hat.shape))


                diff_embedding = close_embedding_beta * F.mse_loss(enc_next_state, encoded_next_state_hat)

                loss = reward_loss + recon_loss + (diff_b * diff_embedding)
                print("t {} Loss {}".format(t, loss.item()))

                loss.backward()
                world_model_optim.step()

    def train(self, init_env, epochs, states, actions, rewards, next_states, dones):
        replay_buffer = self.replay_buffer
        converged = False

        #while not converged:
        for epoch in range(epochs):
            ## Dynamics learning
            self.train_world_model()

            ## Behavior learning
            (imagined_trajectories, imagined_actions, imagined_rewards,
               imagined_values) = self.image_trajectories_rewards_and_values(replay_buffer)

            self.compute_loss_and_update_parameters(imagined_rewards, imagined_values)

            ## Env interaction
            rollout = []
            states_ = []
            actions_ = []
            rewards_ = []
            next_states_ = []
            dones_ = []

            traj_states = []
            traj_actions = []
            traj_rewards = [] 
            traj_next_states = []
            traj_dones = []

            env = deepcopy(init_env)
            obs = env.render('rgb_array')
            obs = cv2.resize(obs, dsize=(resize_, resize_))
            for t in range(self.L):
                traj_states.append(obs)
                obs = torch.unsqueeze(to_tensor(obs), 0)
                z, mu, logvar = self.encoder(obs)

                action = self.actor(z)
                action = torch.argmax(action, 1)
                #action = add_exploration_noise(action)
                traj_actions.append(action)

                obs, reward, done, _ = env.step(action.item())
                traj_rewards.append(reward)

                obs = cv2.resize(obs, dsize=(resize_, resize_))
                traj_next_states.append(obs)

                traj_dones.append(done)
                
            states_.append(traj_states)
            actions_.append(traj_actions)
            rewards_.append(traj_rewards)
            next_states_.append(traj_next_states)
            dones_.append(traj_dones)

            states_ = np.array(states_)
            rewards_ = np.array(rewards_)
            actions_ = np.array(actions_)
            next_states_ = np.array(next_states_)
            
            states = np.concatenate([states, states_], 0)
            actions = np.concatenate([actions, actions_], 0)

            rewards = np.concatenate([rewards, rewards_], 0)
            next_states = np.concatenate([next_states, next_states_], 0)
            
            scaled_states_actions_dataset = MyDatasetSeq(states, (actions, rewards, next_states), transform=to_tensor)
            self.replay_buffer = torch.utils.data.DataLoader(scaled_states_actions_dataset, batch_size=32,
                                                    shuffle=True)


    def image_trajectories_rewards_and_values(self, replay_buffer):
        values = []
        rewards = []
        
        #get all states in replay buffer at time t, i.e. o_t/s_t
        for batch_idx, (state, action, reward, next_state) in enumerate(replay_buffer):

            for t in range(self.L):
                state_t = state[:, t,:, :, :]
                action_t = action[:, t]
                reward_t = reward[:, t]
                next_state_t = next_state[:, t, :, :, :]
                
                z, mu, logvar = self.encoder(state_t)

                # imagine trajectories, starting at time t, i.e s_t (z var in code)
                rollout_rewards = []
                rollout_values = []
                for i in range(self.H):
                    action_hat_one_hot = self.actor(z)
                    action_hat = torch.argmax(action_hat_one_hot, dim=1)
                    action_hat_expanded = torch.unsqueeze(action_hat, 1)

                    reward_t = self.reward_model(z)
                    value_t = self.critic(z)
                    
                    z_expanded = torch.unsqueeze(z, 1)
                    if i == 0:
                        rollout_traj_states = z_expanded
                        rollout_traj_actions = action_hat_expanded
                        rollout_traj_rewards = reward_t
                        rollout_traj_values = value_t
                    else:
                        rollout_traj_states = torch.cat([rollout_traj_states, z_expanded], 1)
                        rollout_traj_actions = torch.cat([rollout_traj_actions, action_hat_expanded], 1)
                        rollout_traj_rewards = torch.cat([rollout_traj_rewards, reward_t], 1)
                        rollout_traj_values = torch.cat([rollout_traj_values, value_t], 1)

                    z = new_state_encoded(z, self.transition_model, action_hat,
                                          self.action_space_size)

                expanded_traj = torch.unsqueeze(rollout_traj_states, 1)
                expanded_traj_actions = torch.unsqueeze(rollout_traj_actions, 1)
                expanded_traj_rewards = torch.unsqueeze(rollout_traj_rewards, 1)
                expanded_traj_values = torch.unsqueeze(rollout_traj_values, 1)
                
                if t == 0 :
                    imagined_trajectories = expanded_traj
                    imagined_actions = expanded_traj_actions
                    imagined_rewards = expanded_traj_rewards
                    imagined_values = expanded_traj_values
                else:
                    imagined_trajectories = torch.cat([imagined_trajectories, expanded_traj], 1)
                    imagined_actions = torch.cat([imagined_actions, expanded_traj_actions], 1)
                    imagined_rewards = torch.cat([imagined_rewards, expanded_traj_rewards], 1)
                    imagined_values = torch.cat([imagined_values, expanded_traj_values], 1)

        return imagined_trajectories, imagined_actions, imagined_rewards, imagined_values
    
    def compute_v_k_N(self, rewards, values, k, tau):
        sum_ = 0
        h = min(tau + k, self.H)
        for n in range(tau, h):
            sum_ += (self.gamma ** (n - tau)) * rewards[:, n - 1]
        sum_ += (self.gamma **(h - tau)) * values[:, h - 1]
        return sum_

    def compute_Vlambda(self, rewards, values):
        # calculate values(objective) for each timestep(tau)
        # in the imagined rollout
        for tau in range(1, self.H + 1):
            sum_= 0
            # sum in V_lambda (exp weighted)
            for n in range(1, self.H):
                coeff = self.lambda_ **(n - 1)
                val = self.compute_v_k_N(rewards, values, n, tau)
                sum_ += coeff * val
            sum_ *= (1 - self.lambda_)
            coeff = self.lambda_ ** (self.H - 1)
            sum_ += coeff * self.compute_v_k_N(rewards, values, self.H, tau)
            sum_expanded = torch.unsqueeze(sum_, 1)

            if tau == 1:
                estimated_values = sum_expanded
            else:
                estimated_values = torch.cat([estimated_values, sum_expanded],1)
        return estimated_values
    
    def compute_loss_and_update_parameters(self, imagined_rewards, imagined_values):
        for t in range(self.L):
            print("t: {}".format(t))
            self.policy_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            rewards = imagined_rewards[:, t, :]
            values = imagined_values[:, t, :]
            est_values = self.compute_Vlambda(rewards, values)

            policy_loss = -torch.sum(est_values)
            critic_loss = self.mse_loss(values, est_values)
            loss = policy_loss + critic_loss
            loss.backward(retain_graph=True)
            print("loss {}".format(loss))

            self.policy_optimizer.step()
            self.critic_optimizer.step()

