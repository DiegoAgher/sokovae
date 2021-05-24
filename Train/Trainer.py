import torch
import torchvision
import matplotlib.pyplot as plt
from IPython import display as displ

from .loss_functions import make_small_objects_important

means = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
stds = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
to_tensor = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                                        mean=means,
                                                        std=stds)
                             ])

def denormalize(normed_img):
    return normed_img * stds[:, None, None] + means[:, None, None]

def show_img(img_tensor):
    img = img_tensor[0]
    denormed_img = denormalize(img)
    plt.imshow(denormed_img.detach().numpy().transpose(1, 2, 0))
    plt.show()

def eval_model_loss(model, loss_fn, loader,
                    non_variational=False, show=False):
    model.eval()
    eval_losses = []

    for batch_idx, (state, action, next_state) in enumerate(loader):
        encoded = model.encoder(state)
        if not non_variational:
            encoded, mu, logvar = model.encoder(state)
        decoded = model.decoder(encoded)
        if batch_idx == 0 and show:
          show_img(state)
          show_img(decoded)
        batch_loss = loss_fn(decoded, state)

        eval_losses.append(batch_loss) 

    return sum(eval_losses) / len(eval_losses)


class Trainer:
    def __init__(self, model,train_loader, eval_loader=None,
                M_N=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.only_train = True if eval_loader is None else False
        if M_N is None:
            sample_batch = _, (state, action, next_state) = next(
                                                             enumerate(self.train_loader)
                                                             )
            sample_batch = state
            M_N = state.shape[0] / len(self.train_loader) 
        self.M_N = M_N 
        self.train_losses = []
        self.kl_losses = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

    def train_one_epoch(self, penalization, epoch_id, loss_function,
                        debug_zeros=False, plot_kl=False, mean_train=False):
        self.model.train()

        mse = torch.nn.MSELoss()
        if loss_function is None and not(loss_function == 'both'):
          loss_function = mse

        loader = self.train_loader
        for batch_idx, (state, action, next_state) in enumerate(loader):
            self.optimizer.zero_grad()

            if debug_zeros:
                z = self.model.encoder(torch.zeros_like(state))
            elif self.model.encoder.non_variational:
                z = self.model.encoder(state)
            else:
                z, mu, logvar = self.model.encoder(state)
                if mean_train:
                    for i in range(9):
                        curr_z, _, _ = ae.encoder(state)
                        z += curr_z
                    z = z / 10

            state_hat = self.model.decoder(z)

            if loss_function == 'both':
              recon_loss_x = mse(state_hat, state)
              small_obj = make_small_objects_important(state_hat, state)
              loss = recon_loss_x + small_obj
            else:
              loss_recon_x = loss_function(state_hat, state)
              loss = loss_recon_x

            if penalization > 0.0:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss *= self.M_N * penalization
                self.kl_losses.append(kl_loss)
                loss_new = kl_loss + loss
                if False: # use_action:
                  action_kl = self.M_N * penalization * action_enc[4].kl_divergence()
                  loss_new += action_kl
            else:
                loss_new = loss

            loss_new.backward()
            self.optimizer.step()

            if (batch_idx % 100 == 0):
                displ.clear_output()
                self.model.eval()
                mse_eval = torch.nn.MSELoss()
                if loss_function == 'both':
                  epoch_train_loss = eval_model_loss(self.model, mse_eval, loader,
                                        non_variational=self.model.encoder.non_variational)
                  if not self.only_train:
                      epoch_eval_loss = eval_model_loss(self.model, mse_eval, loader,
                                        non_variational=self.model.encoder.non_variational)
                  small_obj = make_small_objects_important(state_hat, state)
                else:
                  epoch_train_loss = eval_model_loss(self.model, loss_function, loader,
                                        non_variational=self.model.encoder.non_variational)
                  if not self.only_train:
                      epoch_eval_loss = eval_model_loss(self.model, loss_function,
                                            loader,
                                            non_variational=self.model.encoder.non_variational) 

                _, (state, action, next_state) = next(enumerate(loader))
                show_img(state)
                show_img(state_hat.detach())

                self.train_losses.append(epoch_train_loss.item())
                print("batch {}, epoch {}, loss {}".format(batch_idx, epoch_id,
                                                           epoch_train_loss))

                if self.train_losses[0] / self.train_losses[-1] > 1.75:
                    plt.plot(self.train_losses[-45:])
                else:
                    plt.plot(self.train_losses)
                plt.show()
                if plot_kl:
                    if self.kl_losses[0] / self.kl_losses[-1] > 1.75:
                        plt.plot(self.kl_losses[-45:])
                    else:
                        plt.plot(self.kl_losses)
                    plt.show()
                plt.pause(1)
                self.model.train()

    def set_learning_rate(self, learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def reset_optimizer(self, lr=1e-1):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

