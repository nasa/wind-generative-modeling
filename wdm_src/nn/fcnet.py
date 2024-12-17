from torch import nn
import torch

class FullyConnectedNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layer_dims,
        output_dim,
        activation
    ):
        super().__init__()
        self.layers = nn.Sequential()
        in_dim = input_dim
        if not activation:
            activation = 'RELU'
        for out_dim in hidden_layer_dims:
            self.layers.append(nn.Linear(in_dim, out_dim))
            if activation.upper() == 'RELU':
                self.layers.append(nn.ReLU())
            elif activation.upper() == 'TANH':
                self.layers.append(nn.Tanh())
            else:
                raise Exception("Invalid activation function")
            in_dim = out_dim

        self.layers.append(nn.Linear(in_dim, output_dim))

    def forward(self, x_t, time):
        time = time.reshape((len(time), 1))
        x = torch.cat((time, x_t), 1)
        x = self.layers(x)
        return x
