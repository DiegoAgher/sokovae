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

resize_ = 40

def new_state_encoded(z, action_model, action_onehot, num_classes):
    cond_action = torch.cat([z, action_onehot], 1)
    encoded_action = action_model(cond_action).type(torch.float)
    return z + encoded_action


class Dreamer(torch.nn.Module):
    def __init__(self, encoder, decoder, action_space_size, latent_size=11,
                 gamma=0.99, horizon=12, L=15, data_replay_buffer=None, lambda_=0.95):
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
        self.data_replay_buffer = data_replay_buffer
        (self.states, self.actions,
          self.rewards, self.next_states) = self.data_replay_buffer
        self.replay_buffer = create_loader((self.states, self.actions, self.rewards, self.next_states))

        self.actor = self.build_mlp(input_size=self.latent_size, output_size=action_space_size)
        self.actor_params = [param for param in self.actor.parameters()]
        self.actor_optimizer = optim.Adam(self.actor_params, lr=1e-1)

        self.critic = self.build_mlp(input_size=self.latent_size)
        self.critic_params = [param for param in self.critic.parameters()]
        self.critic_optimizer = optim.Adam(self.critic_params, lr=1e-3)
        self.mse_loss = nn.MSELoss()

    def build_world_model_optim(self):
        world_model_params = [x for x in self.encoder.parameters()] + [x for x in self.decoder.parameters()]
        world_model_params = world_model_params + [x for x in self.reward_model.parameters()]
        world_model_params = world_model_params + [x for x in self.transition_model.parameters()]
        self.world_model_params = world_model_params
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
        mse_reward = nn.MSELoss()
        loader = self.replay_buffer
        world_model_optim = self.world_model_optim

        for batch_idx, (state, action, reward, next_state) in enumerate(loader):

            world_model_optim.zero_grad()

            for t in range(self.L):
                state_t = state[:, t,:, :, :]
                action_t = action[:, t]
                action_onehot_t = action_onehot = F.one_hot(action_t,
                                                            num_classes=self.action_space_size).type(torch.float)
                reward_t = reward[:, t].float()
                next_state_t = next_state[:, t, :, :, :]

                z, mu, logvar = self.encoder(state_t)

                encoded_next_state_hat = new_state_encoded(z, self.transition_model,
                                                          action_onehot_t, self.action_space_size)
                next_state_hat = self.decoder(encoded_next_state_hat)
                recon_loss = mse(next_state_hat, next_state_t)


                reward_hat = self.reward_model(z)
                reward_loss = mse_reward(reward_t, reward_hat)

                # remains to implement kl loss
                # kl_loss = torch.distributions.kl.kl_divergence(z, next_state_hat)
                enc_next_state, mu, logvar = self.encoder(next_state_t)
                #enc_next_state = torch.flatten(enc_next_state)

                encoded_next_state_hat = torch.reshape(encoded_next_state_hat,
                                                       enc_next_state.shape)

                diff_embedding = close_embedding_beta * F.mse_loss(enc_next_state, encoded_next_state_hat)

                loss = reward_loss + recon_loss + (diff_b * diff_embedding)

                loss.backward()
                world_model_optim.step()
        print("t {} WorldLoss {}".format(t, loss.item()))
        print("reward loss {}, recon {}".format(reward_loss.item(),
                                                        recon_loss.item()))
        print(" ")

    def train(self, init_env, epochs):
        replay_buffer = self.replay_buffer
        converged = False

        # while not converged:
        for epoch in range(epochs):
            print("####")
            print("starting epoch {}".format(epoch))
            print("####")
            ## Dynamics learning
            self.train_world_model()

            ## Behavior
            for batch_id, buffer_data in enumerate(replay_buffer):
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                with ParamsNoGrad(self.world_model_params):
                    (imagined_states,
                    imagined_actions) = self.imagine_trajectories(buffer_data)

                # predict rewards and values
                with ParamsNoGrad(self.world_model_params + self.critic_params):
                    (predicted_rewards,
                    predicted_values) = (self.reward_model(imagined_states),
                                         self.critic(imagined_states))

                # calculate returns for actor
                discount_arr = self.gamma * torch.ones_like(predicted_rewards)
                returns = compute_return(predicted_rewards[:-1], predicted_values[:-1],
                                         discount_arr[:-1], bootstrap=predicted_values[-1],
                                         lambda_=self.lambda_)
                discount_arr = torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[1:]])
                discount = torch.cumprod(discount_arr[:-1], 0)
                actor_loss = -torch.mean(discount * returns)

                # calculate values for critic
                mse_ = torch.nn.MSELoss()
                with torch.no_grad():
                    value_feat = imagined_states[:-1].detach()
                    value_discount = discount.detach()
                    value_target = returns.detach()
                critic_pred = dreamer.critic(value_feat)
                critic_loss = mse_(value_target, critic_pred)

                loss = actor_loss + critic_loss
                print("batch_id {}, actor loss {}, critic loss {}".format(
                        batch_id, actor_loss.item(), critic_loss.item()))
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

            ## ENV INTERACTION
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

            env = deepcopy(close_init_env)
            obs = env.render('rgb_array')
            obs = cv2.resize(obs, dsize=(resize_, resize_))

            with torch.no_grad():
                for t in range(self.L):
                    traj_states.append(obs)
                    obs = torch.unsqueeze(to_tensor(obs), 0)
                    z, mu, logvar = self.encoder(obs)

                    action_probs = F.softmax(self.actor(z))
                    dist = torch.distributions.categorical.Categorical(action_probs)
                    action = dist.sample().item()
                    #action = add_exploration_noise(action)
                    traj_actions.append(action)

                    obs, reward, done, _ = env.step(action)
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

            self.states = np.concatenate([self.states, states_], 0)
            self.actions = np.concatenate([self.actions, actions_], 0)

            self.rewards = np.concatenate([self.rewards, rewards_], 0)
            self.next_states = np.concatenate([self.next_states, next_states_], 0)

            self.replay_buffer = create_loader((self.states, self.actions, self.rewards, self.next_states))


    def imagine_trajectories(self, buffer_data):
        #get all states in replay buffer at time t, i.e. o_t/s_t
        obs, _, _, _ = buffer_data
        batch_imagined_states = []
        batch_imagined_actions = []
        for tau in range(self.L):
            tau = 0
            obs_tau = obs[:, tau, :, :, :]

            with ParamsNoGrad(self.world_model_params):
                state_tau, mu, logvar = self.encoder(obs_tau)


            with ParamsNoGrad(self.world_model_params):
                actions = []
                states = []
                # dream for H(horizon) timesteps
                for k in range(self.H):
                    action_onehot = self.actor(state_tau)
                    actions.append(torch.argmax(action_onehot, dim=1))

                    state_tau = new_state_encoded(state_tau, self.transition_model,
                                                  action_onehot, self.action_space_size)
                    states.append(state_tau)

                states = torch.stack(states, 1)
                actions = torch.stack(actions, 1)
            batch_imagined_states.append(states)
            batch_imagined_actions.append(actions)

        batch_imagined_states = torch.stack(batch_imagined_states, 0)
        batch_imagined_states = torch.flatten(batch_imagined_states, 1,2)

        batch_imagined_actions = torch.stack(batch_imagined_actions, 0)
        batch_imagined_actions = torch.flatten(batch_imagined_actions, 1,2)

        return batch_imagined_states, batch_imagined_actions

    def solve(self, env, max_steps=100):
        env = deepcopy(close_init_env)
        obs = env.render('rgb_array')
        done = False
        steps = 0
        ep_return = 0
        with torch.no_grad():
            while (not done) and steps < max_steps:
                obs = cv2.resize(obs, dsize=(resize_, resize_))
                obs = torch.unsqueeze(to_tensor(obs), 0)
                z, mu, logvar = self.encoder(obs)

                action_probs = F.softmax(self.actor(z))
                action = torch.argmax(action_probs, dim=1).item()

                obs, reward, done, _ = env.step(action)
                ep_return += reward
                steps += 1
        print("episode return {}".format(ep_return))

    def compute_v_k_N(self, rewards, values, k, tau):
        sum_ = 0
        h = min(tau + k, self.H)
        for n in range(tau, h):
            sum_ = sum_ + (self.gamma ** (n - tau)) * rewards[:, n - 1]
        sum_ = sum_ + (self.gamma **(h - tau)) * values[:, h - 1]
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
                sum_ = sum_ + coeff * val.clone()
            sum_ = sum_ * (1 - self.lambda_)
            coeff = self.lambda_ ** (self.H - 1)
            sum_ = sum_ + coeff * self.compute_v_k_N(rewards, values, self.H, tau).clone()
            sum_expanded = sum_.unsqueeze(1)

            if tau == 1:
                estimated_values = sum_expanded
            else:
                estimated_values = torch.cat([estimated_values, sum_expanded],1)
        return estimated_values
