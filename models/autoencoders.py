import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.nn import LeakyReLU, ReLU, Conv2d, ConvTranspose2d, Tanh
from torch.nn import BatchNorm2d, Dropout, Dropout2d, Flatten

from .bayeslayers import BayesianLayer

class BaseVariational(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, debug=False):
        for layer in self.layers:
            x = layer(x)
            # if isinstance(layer, nn.Flatten):
            #     print(x.shape)
            if debug:
                print("layer {}".format(layer))
                print("shape {}".format(x.shape))
                print("")
        return x

    def predict(self, x, num_forward_passes=10):
        pred = self.forward(x)
        for i in range(num_forward_passes - 1):
            pred += self.forward(x)
        pred = pred / num_forward_passes

        return latent
   
    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        kl = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BayesianLayer):
                kl_ = layer.kl_divergence()
                kl += kl_
        return kl

class DeepEncoderSmall(BaseVariational):
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
                Conv2d(in_channels=256, out_channels=256,
                          kernel_size=3, stride=3, padding=4),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                ReLU(),
                Conv2d(in_channels=512, out_channels=512,
                          kernel_size=3, stride=3, padding=4),
                BatchNorm2d(512),
                ReLU(),
                Conv2d(in_channels=512, out_channels=1024,
                          kernel_size=3, stride=1, padding=1),
                BatchNorm2d(1024),
                ReLU(),
                Conv2d(in_channels=1024, out_channels=1024,
                          kernel_size=3, stride=3, padding=4),
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

class DeepDecoderSmall(BaseVariational):
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
                # ConvTranspose2d(in_channels=1024, out_channels=1024,
                #           kernel_size=3, stride=3, padding=3),
                # BatchNorm2d(1024),
                # Dropout2d(p=0.25),
                # ReLU(),
                ConvTranspose2d(in_channels=1024, out_channels=1024,
                          kernel_size=3, stride=3, padding=3, output_padding=1),
                BatchNorm2d(1024),
                Dropout2d(p=0.25),
                ReLU(),
                ConvTranspose2d(in_channels=1024, out_channels=512,
                          kernel_size=3, stride=2, padding=1),
                BatchNorm2d(512),
                Dropout2d(p=0.25),
                ReLU(),
                # ConvTranspose2d(in_channels=512, out_channels=512,
                #           kernel_size=3, stride=3, padding=3),
                # BatchNorm2d(512),
                # Dropout2d(p=0.25),
                # ReLU(),
                ConvTranspose2d(in_channels=512, out_channels=256,
                                kernel_size=3, stride=2, padding=3),
                BatchNorm2d(256),
                ReLU(),
                ConvTranspose2d(in_channels=256, out_channels=256,
                          kernel_size=3, stride=2, padding=3),
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
                          kernel_size=3, stride=1, padding=2),
                BatchNorm2d(32),
                ReLU(),
                ConvTranspose2d(in_channels=32, out_channels=16,
                                          kernel_size=3, stride=1, padding=2),
                ReLU(),
                BatchNorm2d(16),
                Conv2d(in_channels=16, out_channels=8,
                          kernel_size=4, stride=1, padding=1),
                ReLU(),
                Conv2d(in_channels=8, out_channels=3,
                          kernel_size=3, stride=1, padding=0),
                Tanh()
            ]
        return layers

    def forward(self, x, debug=False):
        x = self.start_decode(x)
        x = self.drop_linear_one(x)
        x = F.relu(x)
        x = self.start_decodeb(x)
        x = self.drop_linear_two(x)
        x = F.relu(x)
        x = x.reshape(-1, 1024, 3, 3)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if debug:
                print("layer {}".format(layer))
                print("shape {}".format(x.shape))
                print("")
        return x


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

    def forward(self, x, debug=False):
        x = self.start_decode(x)
        x = self.drop_linear_one(x)
        x = F.relu(x)
        x = self.start_decodeb(x)
        x = self.drop_linear_two(x)
        x = F.relu(x)
        x = x.reshape(-1, 1024, 3, 3)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if debug:
                print("layer {}".format(layer))
                print("shape {}".format(x.shape))
                print("")
        return x

class CnnAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstruction = self.decoder(latent_representation)
        return reconstruction

    def predict(self, x, num_forward_passes=10):

        latent_representation = self.encoder(x)
        reconstruction = self.decoder(latent_representation)

        for i in range(num_forward_passes - 1):
            curr_latent = self.encoder(x)
            latent_representation += curr_latent
            reconstruction += self.decoder(curr_latent)
        latent_representation = latent_representation / num_forward_passes
        reconstruction = reconstruction / num_forward_passes

        return latent_representation, reconstruction
    def params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        kl = 0
        for i, layer in enumerate(self.encoder.layers):
            if isinstance(layer, BayesianLayer):
                kl_ = layer.kl_divergence()
                kl += kl_
        return kl



class CondCnnAE(torch.nn.Module):
    def __init__(self, encoder, action_encoder, decoder):
        super(CondCnnAE, self).__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.decoder = decoder
        

    def forward(self, x, action):
        encoded_x = self.encoder(x)
        encoded_action = self.action_encoder(action)
        encoded_blend = encoded_x + encoded_action 
        reconstruction = self.decoder(encoded_blend)
        return reconstruction

    def predict(self, x, num_forward_passes=10):

        # TODO: make n random forward passes
        encoded_x = self.encoder(x)
        encoded_action = self.action_encoder(action)
        encoded_blend = x + encoded_action
        reconstruction = self.decoder(encoded_blend)

        for i in range(num_forward_passes - 1):
            curr_enc_x = self.encoder(x)
            encoded_x += curr_enc_x

            curr_enc_a = self.action_encoder(action)
            #TODO if encoding of action is VAE then do M samples
            
            curr_blend = curr_enc_x + curr_enc_a

            reconstruction += self.decoder(curr_blend)

        encoded_x = encoded_x / num_forward_passes
        reconstruction = reconstruction / num_forward_passes

        return encoded_x, reconstruction
    
    def params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())


    def kl_loss(self):
        '''
        Computes the KL divergence loss for all layers.
        '''
        # TODO: enter your code here
        kl = 0
        for i, layer in enumerate(self.encoder.layers):
            if isinstance(layer, BayesianLayer):
                kl_ = layer.kl_divergence()
                kl += kl_
        return kl

class SmallEncoder40(BaseVariational):
    '''
    Takes in as input batches of images of size [batch_size, 40, 40, 3]
    '''
    def __init__(self, latent_size, layers_dims, activation='relu'):
        super().__init__()
        self.latent_size = latent_size
        self.layers_dims = layers_dims
        self.activation = LeakyReLU if activation is not 'relu' else ReLU
        self.layers = torch.nn.ModuleList(self._init_layers())

    def _init_layers(self):
        layers = []
        for idx, dims in enumerate(self.layers_dims):
            if idx == 0:
                current_dims = 3
                next_dims = dims
            else:
                current_dims = self.layers_dims[idx - 1]
                next_dims = dims

            layers.extend(
                    [
                    Conv2d(in_channels=current_dims, out_channels=next_dims,
                           kernel_size=3, stride=2, padding=1),
                    BatchNorm2d(next_dims),
                    self.activation(),
                    ])

        flat_size = self.layers_dims[-1] * 5 * 5
        layers.extend([
                Flatten(),
                Dropout(p=0.25),
                Linear(flat_size, int(flat_size / 9)),
                ReLU(),
                #Dropout(p=0.4),
                BayesianLayer(int(flat_size / 9), self.latent_size)]
        )
        return layers
class SmallDecoder40(BaseVariational):
    '''
    Use with SmallEncoder40.
    will take a Dense vector and turn it into batches of size [batch_size, 40, 40, 3]
    '''
    def __init__(self, latent_size, layers_dims, activation='relu'):
        super().__init__()
        self.latent_size = latent_size
        self.layers_dims = layers_dims
        self.activation = LeakyReLU if activation is not 'relu' else ReLU
        self.layers = torch.nn.ModuleList(self._init_layers())

    def _init_layers(self):
        flat_size = self.layers_dims[0] * 5 * 5
        self.start_decode = Linear(self.latent_size, int(flat_size / 9))
        self.drop_linear_one = Dropout(p=0.4)
        self.start_decodeb = Linear(int(flat_size / 9), flat_size)
        self.drop_linear_two = Dropout(p=0.4)

        stride = 3
        padding = 3
        output_padding = 0
        layers = []
        for idx, dims in enumerate(self.layers_dims):

            if idx == 2:
                current_dims = dims
                next_dims = 8
            else:
                current_dims = dims
                next_dims = self.layers_dims[idx + 1]

            if idx == 2:
                stride = 2
                padding = 1
                output_padding = 1

            layers.extend(
                [
                Dropout2d(p=0.1),
                ConvTranspose2d(in_channels=current_dims,
                                 out_channels=next_dims,
                                 kernel_size=3, stride=stride, padding=padding,
                                 output_padding=output_padding),
                BatchNorm2d(next_dims),
                self.activation()
                ]
              )

        layers.extend(
                [Conv2d(in_channels=8, out_channels=3,
                        kernel_size=3, stride=1, padding=0),
                 Tanh()]
                    )
        return layers
    
    def forward(self, x, debug=False):
        x = self.drop_linear_one(x)
        x = self.start_decode(x)
        x = F.relu(x)
        x = self.drop_linear_two(x)
        x = self.start_decodeb(x)
        x = F.relu(x)
        
        init_channels = self.layers_dims[0]
        x = x.reshape(-1, init_channels, 5, 5)

        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if debug:
                print("layer {}".format(layer))
                print("shape {}".format(x.shape))
                print("")
        return x
