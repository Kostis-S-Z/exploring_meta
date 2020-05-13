#!/usr/bin/env python3

"""
Taken directly from https://github.com/learnables/learn2learn/tree/master/examples/rl
"""

import math

import cherry as ch
import torch
from torch import nn
from torch.distributions import Normal, Categorical

EPSILON = 1e-6


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu'):
        super(DiagNormalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class ANILDiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, fc_neurons, hiddens=None):
        super(ANILDiagNormalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        activation = nn.Tanh
        self.fc_neurons = fc_neurons

        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())

        # Initialize the body of the network
        self.features = nn.Sequential(*layers)

        # Initialize the head of the network
        self.head = linear_init(nn.Linear(fc_neurons, output_size))
        self.sigma = torch.nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))
        # Do not store gradients for the network body when doing a forward pass
        self.features_no_grad = False

    def turn_on_body_grads(self):
        self.features_no_grad = False

    def turn_off_body_grads(self):
        self.features_no_grad = True

    def forward_pass(self, state):
        if self.features_no_grad:
            with torch.no_grad():
                state_features = self.features(state)
        else:
            state_features = self.features(state)
        return self.head(state_features)

    def density(self, state):
        loc = self.forward_pass(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class DiagNormalPolicyCNN(nn.Module):

    def __init__(self, input_size, output_size, network=[32, 64, 64]):
        super(DiagNormalPolicyCNN, self).__init__()

        n_layers = len(network)
        activation = nn.ReLU

        # Building a network using a dictionary this way ONLY THIS ONLY WORKS FOR PYTHON 3.7
        # Otherwise the dictionary won't remember the order!
        # Define input layer
        features = {"conv_0": nn.Conv2d(in_channels=input_size, out_channels=network[0], kernel_size=3, padding=1),
                    "bn_0": nn.BatchNorm2d(network[0]),
                    "activation_0": activation(),
                    "max_pool_0": nn.MaxPool2d(kernel_size=2, stride=2)}

        # Initialize weights of input layer
        maml_init_(features["conv_0"])
        nn.init.uniform_(features["bn_0"].weight)

        # Define rest of hidden layers and initialize their weights
        for i in range(1, n_layers):
            layer_i = {f"conv_{i}": nn.Conv2d(in_channels=network[i - 1], out_channels=network[i],
                                              kernel_size=3, stride=1, padding=1),
                       f"bn_{i}": nn.BatchNorm2d(network[i]),
                       f"activation_{i}": activation(),
                       f"max_pool_{i}": nn.MaxPool2d(kernel_size=2, stride=2)}

            maml_init_(layer_i[f"conv_{i}"])
            nn.init.uniform_(layer_i[f"bn_{i}"].weight)
            features.update(layer_i)

        # Given a 64x64 pixel calculate the flatten size needed based on the depth of the network
        # and how "fast" (=stride) it downscales the image
        final_pixel_dim = int(64 / (math.pow(2, n_layers)))
        self.flatten_size = network[-1] * final_pixel_dim * final_pixel_dim

        head = nn.Linear(in_features=self.flatten_size, out_features=output_size, bias=True)  # No activation for output
        maml_init_(head)

        self.features = nn.Sequential(*list(features.values()))
        self.mean = head
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))
        # This is just a trivial assignment to follow the implementation of the sampler
        self.step = self.forward

    def density(self, state):
        # Pass images through CNN to get features
        state = self.features(state)
        # Flatten features to 1-dim for the FC layer
        state = state.view(-1, self.flatten_size)
        # Pass features to the FC output layer
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class BaselineCNN(nn.Module):
    def __init__(self, input_size, network=[32, 64, 64]):
        super(BaselineCNN, self).__init__()

        n_layers = len(network)
        activation = nn.ReLU

        # Building a network using a dictionary this way ONLY THIS ONLY WORKS FOR PYTHON 3.7
        # Otherwise the dictionary won't remember the order!
        # Define input layer
        features = {"conv_0": nn.Conv2d(in_channels=input_size, out_channels=network[0], kernel_size=3, padding=1),
                    "bn_0": nn.BatchNorm2d(network[0]),
                    "activation_0": activation(),
                    "max_pool_0": nn.MaxPool2d(kernel_size=2, stride=2)}

        # Initialize weights of input layer
        maml_init_(features["conv_0"])
        nn.init.uniform_(features["bn_0"].weight)

        # Define rest of hidden layers and initialize their weights
        for i in range(1, n_layers):
            layer_i = {f"conv_{i}": nn.Conv2d(in_channels=network[i - 1], out_channels=network[i],
                                              kernel_size=3, stride=1, padding=1),
                       f"bn_{i}": nn.BatchNorm2d(network[i]),
                       f"activation_{i}": activation(),
                       f"max_pool_{i}": nn.MaxPool2d(kernel_size=2, stride=2)}

            maml_init_(layer_i[f"conv_{i}"])
            nn.init.uniform_(layer_i[f"bn_{i}"].weight)
            features.update(layer_i)

        # Given a 64x64 pixel calculate the flatten size needed based on the depth of the network
        # and how "fast" (=stride) it downscales the image
        final_pixel_dim = int(64 / (math.pow(2, n_layers)))
        self.flatten_size = network[-1] * final_pixel_dim * final_pixel_dim

        head = nn.Linear(in_features=self.flatten_size, out_features=1, bias=True)  # No activation for output
        linear_init(head)

        self.features = nn.Sequential(*list(features.values()))
        self.head = head

    def forward(self, state):
        # Pass images through CNN to get features
        features = self.features(state)
        # Flatten features to 1-dim for the FC layer
        features = features.view(-1, self.flatten_size)
        # Pass features to the FC output layer
        value = self.head(features)
        return value


class CategoricalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None):
        super(CategoricalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, state):
        state = ch.onehot(state, dim=self.input_size)
        loc = self.mean(state)
        density = Categorical(logits=loc)
        action = density.sample()
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return action, {'density': density, 'log_prob': log_prob}
