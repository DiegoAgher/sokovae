import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.nn import ReLU, Conv2d, ConvTranspose2d, Tanh
from torch.nn import BatchNorm2d, Dropout, Dropout2d, Flatten

from .MyLayers import BayesianLayer


class BaseVariational(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # if isinstance(layer, nn.Flatten):
            #     print(x.shape)
            # print("layer {}".format(layer))
            # print("shape {}".format(x.shape))
            # print("")
        return x

    def predict(self, x, num_forward_passes=10):
        batch_size = x.shape[0]

        # TODO: make n random forward passes
        latent = self.forward(x)
        for i in range(num_forward_passes - 1):
            latent += self.forward(x)
        latent = latent / num_forward_passes

        return latent
    
    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        # TODO: enter your code here
        kl = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianLayer):
                kl_ = layer.kl_divergence()
                kl += kl_
        return kl


class EncoderSmall(BaseVariational):
    '''
    Takes in as input batches of images of size [batch_size, 40, 40, 3]
    '''
    def __init__(self, latent_size=9):
        super().__init__()
        self.latent_size = latent_size
        self.layers = torch.nn.ModuleList(self._init_layers())

    def _init_layers(self):
        layers = [
                Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
                ReLU(),
                Conv2d(in_channels=32, out_channels=64,
                          kernel_size=3, stride=2, padding=1),
                ReLU(),
                Conv2d(in_channels=64, out_channels=128,
                          kernel_size=3, stride=2, padding=1),
                BatchNorm2d(128),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256,
                          kernel_size=3, stride=2, padding=1),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                ReLU(),
                Conv2d(in_channels=512, out_channels=1024,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(1024),
                ReLU(),
                Dropout2d(p=0.25),
                Flatten(),
                Dropout(p=0.4),
                Linear(9216, int(9216 / 9)),
                ReLU(),
                Dropout(p=0.4),
                BayesianLayer(int(9216 / 9), self.latent_size)]
        return layers


class DecoderSmall(BaseVariational):
    '''
    Use with SmallDecoder.
    will take a Dense vector and turn it into batches of size [batch_size, 40, 40, 3]
    '''
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.layers = torch.nn.ModuleList(self._init_layers())

    def _init_layers(self):
        self.start_decode = Linear(self.latent_size, int(9216 / 9))
        self.drop_linear_one = Dropout(p=0.4)
        self.start_decodeb = Linear(int(9216 / 9), 9216)
        self.drop_linear_two = Dropout(p=0.4)
        layers = [
                ConvTranspose2d(in_channels=1024, out_channels=512,
                          kernel_size=3, stride=2, padding=1, output_padding=1),
                BatchNorm2d(512),
                Dropout2d(p=0.25),
                ReLU(),
                ConvTranspose2d(in_channels=512, out_channels=256,
                          kernel_size=3, stride=2, padding=1, output_padding=1),
                BatchNorm2d(256),
                ReLU(),
                ConvTranspose2d(in_channels=256, out_channels=128,
                          kernel_size=3, stride=2, padding=1),
                BatchNorm2d(128),
                ReLU(),
                ConvTranspose2d(in_channels=128, out_channels=64,
                      kernel_size=3, stride=2, padding=2),
                BatchNorm2d(64),
                ReLU(),
                ConvTranspose2d(in_channels=64, out_channels=32,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(32),
                ReLU(),
                ConvTranspose2d(in_channels=32, out_channels=16,
                          kernel_size=3, stride=1, padding=1),
                ReLU(),
                BatchNorm2d(16),
                # ConvTranspose2d(in_channels=16, out_channels=16,
                #           kernel_size=3, stride=1, padding=1),
                # BatchNorm2d(16),
                # ReLU(),
                Conv2d(in_channels=16, out_channels=8,
                          kernel_size=4, stride=1, padding=1),
                ReLU(),
                Conv2d(in_channels=8, out_channels=3,
                          kernel_size=3, stride=1, padding=0),
                Tanh()
            ]
        return layers

