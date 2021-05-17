import torch
from torch import nn


def sample_reparam(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample

class BayesianLayer(torch.nn.Module):
    '''
    Module implementing a single Bayesian feedforward layer.
    The module performs Bayes-by-backprop, that is, mean-field
    variational inference. It keeps prior and posterior weights
    (and biases) and uses the reparameterization trick for sampling.
    '''
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias

        self.prior_mu = 0
        self.prior_sigma = 1
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim),
                                      requires_grad=True)
        self.weight_logsigma = nn.Parameter(torch.Tensor(output_dim, input_dim),
                                           requires_grad=True)
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.uniform_(self.weight_logsigma, -5,-4)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(output_dim),
                                            requires_grad=True)
            self.bias_logsigma = nn.Parameter(torch.Tensor(output_dim),
                                             requires_grad=True)
            nn.init.constant_(self.bias_mu, 0.0)
            nn.init.uniform_(self.bias_logsigma, -5,-4)

        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)

        #self.reset_parameters()

    def reset_parameters(self) -> None:
        # Linear Layer torch implementation
        init.kaiming_uniform_(self.weight_logsigma, a=np.sqrt(5))
        init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        if self.use_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_logsigma)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias_mu, -bound, bound)
            init.uniform_(self.bias_logsigma, -bound, bound)

    def forward(self, inputs):
        weights = sample_reparam(self.weight_mu, self.weight_logsigma)

        if self.use_bias:
            bias = sample_reparam(self.bias_mu, self.bias_logsigma)
            return torch.nn.functional.linear(inputs, weights, bias)

        return torch.nn.functional.linear(inputs, weights)

    def kl_divergence(self):
        '''
        Computes the KL divergence between the priors and posteriors for this layer.
        '''
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss += self._kl_divergence(self.bias_mu, self.bias_logsigma)
        return kl_loss
    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''

        logsigma_flat = torch.flatten(logsigma)
        mu_flat = torch.flatten(mu)

        std_flat = torch.exp(0.5*logsigma_flat)
        var_flat = torch.exp(logsigma_flat)


        var_prior = self.prior_sigma
        mean_prior = self.prior_mu

        # num_denom = ( (var_a_flat) + (mean_prior - mu_flat)**2 ) / var_prior
        # kl = (torch.ones_like(mu_flat) - num_denom
        #         + logvar_flat - torch.zeros_like(logvar_flat)).mean()

        #kl = (var_flat + (mu_flat**2) - logsigma_flat - torch.ones_like(logsigma_flat)).mean()
        kl = ( torch.log(self.prior_sigma/std_flat) +
               ( (var_flat  + (mu_flat - self.prior_mu)**2) / 2 * self.prior_sigma) -
                 0.5).mean()
        return kl

