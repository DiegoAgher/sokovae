import torch
import matplotlib.pyplot as plt


def denormalize(normed_img):
    return normed_img * stds[:, None, None] + means[:, None, None]

def show_img(img_tensor):
    img = img_tensor[0]
    denormed_img = denormalize(img)
    plt.imshow(denormed_img.detach().numpy().transpose(1, 2, 0))
    plt.show()

class Trainer:
    def __init__(self, model,train_loader, eval_loader=None):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.only_train = True if eval_loader is None else False
        self.train_losses = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

    def train_one_epoch(self, penalization, model, epoch_id, loss_function):
        self.model.train()

        mse = torch.nn.MSELoss()
        if loss_function is None and not(loss_function == 'both'):
          loss_function = mse

        loader = self.train_loader
        for batch_idx, (state, action, next_state) in enumerate(loader):
            optimizer.zero_grad()

            z = self.model.encoder(state)
            state_hat = self.model.decoder(z)
            if loss_function == 'both':
              recon_loss_x = mse(state_hat, state)
              small_obj = make_small_objects_important(state_hat, state)
              loss = recon_loss_x + small_obj
            else:
              loss_recon_x = loss_function(state_hat, state)
              loss = loss_recon_x

            if penalization > 0.0:
                kl_loss = M_N * penalization * self.model.kl_loss()
                loss_new = kl_loss + loss
                if use_action:
                  action_kl = M_N * penalization * action_enc[4].kl_divergence()
                  loss_new += action_kl
            else:
                loss_new = loss

            loss_new.backward()
            self.optimizer.step()

            if (batch_idx % 100 == 0):
                displ.clear_output()
                self.model.eval()
                mse_eval = nn.MSELoss()
                if loss_function == 'both':
                  epoch_train_loss = eval_model_loss(model, mse_eval, True ,'train')
                  if not self.only_train:
                      epoch_eval_loss = eval_model_loss(model, mse_eval, True, 'eval')
                  small_obj = make_small_objects_important(state_hat, state)
                else:
                  epoch_train_loss = eval_model_loss(model, loss_function, True ,'train')
                  if not self.only_train:
                      epoch_eval_loss = eval_model_loss(model, loss_function, True, 'eval')

                _, (state, action, next_state) = next(enumerate(self.loader))
                show_img(state)
                show_img(state_hat.detach())

                self.train_losses.append(epoch_train_loss.item())

                if self.train_losses[0] / self.train_losses[-1] > 1.75:
                    plt.plot(self.train_losses[-45:])
                else:
                    plt.plot(self.train_losses)
                plt.show()
                self.model.train()

    def set_learning_rate(self, learning_rate):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate

    def reset_optimizer(self, lr=1e-1):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

