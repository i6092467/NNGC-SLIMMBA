"""
Modules for inferring Granger causality based on multilayer perceptrons (MLPs).
"""
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class MLPgc(nn.Module):
    # This class defines the MLP model for inferring Granger causality in a multivariate time series.
    def __init__(self, num_vars, device, lag, hidden_size_1, hidden_size_2, num_outputs=1, dp=0.0):
        """
        Initialises an MLPgc module, which represents a neural network model for Granger causality estimation.

        :param num_vars: number of variables, including the response.
        :param device: device to be used for calculations, CPU or GPU.
        :param lag: order of considered regressive relationships, specifies the horizon in the past of predictors to be
        used to forecast the future of the response.
        :param hidden_size_1: size of layers 1 and 2 in sub-networks.
        :param hidden_size_2: size of layer 3.# Contains various utility functions for construct GC models and generating synthetic time series.
        :param num_outputs: number of output units.
        :param dp: dropout rate applied to all layers, to prevent the co-adaptation of neurons. Default value 0.0, i.e.
        no dropout.
        """
        super(MLPgc, self).__init__()

        # Sub-networks
        self.layer1_list = nn.ModuleList()
        self.layer2_list = nn.ModuleList()
        for state in range(num_vars):
            layer1 = nn.Linear(lag, hidden_size_1)
            layer2 = nn.Linear(hidden_size_1, hidden_size_1)
            self.layer1_list.append(layer1)
            self.layer2_list.append(layer2)

        # Initialise weights for each variable
        self.imp_weights = nn.Parameter(torch.Tensor(np.ones((num_vars, )) / num_vars +
                                                     np.random.normal(0, 0.00001, (num_vars, ))).float().to(device))

        # Final layers
        self.layer_3 = nn.Linear(hidden_size_1 * num_vars, hidden_size_2)
        self.layer_4 = nn.Linear(hidden_size_2, num_outputs)

        # Initialise the rest of the weights
        self.init_weights()

        # Save parameters
        self.num_vars = num_vars
        self.lag = lag
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dp = dp
        self.device = device

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # Forward propagation
    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: inputs with shape [batch size, lag * number of variables]
        :return: returns the forecast for the future value of the target variable.
        """
        aggregated = None

        # Propagate in sub-networks
        for i in range(self.num_vars):
            layer_1 = self.layer1_list[i]
            layer_2 = self.layer2_list[i]
            inp = inputs[:, (self.lag * i):(self.lag * (i + 1))]
            tmp = F.dropout(F.relu(layer_2(F.dropout(F.relu(layer_1(inp)), p=self.dp, training=True))),
                            p=self.dp, training=True)
            if i == 0:
                aggregated = self.imp_weights[i] * tmp
            else:
                aggregated = torch.cat((aggregated, self.imp_weights[i] * tmp), dim=1)

        # Final two layers
        pred = self.layer_4(F.dropout(F.relu(self.layer_3(aggregated)), p=self.dp, training=True))

        return pred


class MLPgcLinear(nn.Module):
    # This class defines the MLP model for inferring Granger causality in a multivariate time series.
    # This module has only linear activation functions.
    def __init__(self, num_vars, device, lag, hidden_size_1, hidden_size_2, num_outputs=1, dp=0.0):
        """
        Initialises an MLPgc module with linear activation functions, which represents a neural network model for
        Granger causality estimation.

        :param num_vars: number of variables, including the response.
        :param device: device to be used for calculations, CPU or GPU.
        :param lag: order of considered regressive relationships, specifies the horizon in the past of predictors to be
        used to forecast the future of the response.
        :param hidden_size_1: size of layers 1 and 2 in sub-networks.
        :param hidden_size_2: size of layer 3.
        :param num_outputs: number of output units.
        :param dp: dropout rate applied to all layers, to prevent the co-adaptation of neurons. Default value 0.0, i.e.
        no dropout.
        """
        super(MLPgcLinear, self).__init__()

        # Sub-networks
        self.layer1_list = nn.ModuleList()
        self.layer2_list = nn.ModuleList()
        for state in range(num_vars):
            layer1 = nn.Linear(lag, hidden_size_1)
            layer2 = nn.Linear(hidden_size_1, hidden_size_1)
            self.layer1_list.append(layer1)
            self.layer2_list.append(layer2)

        # Initialise weights for each variable
        self.imp_weights = nn.Parameter(torch.Tensor(np.ones((num_vars,)) / num_vars +
                                                     np.random.normal(0, 0.00001, (num_vars,))).float().to(device))

        # Final layers
        self.layer_3 = nn.Linear(hidden_size_1 * num_vars, hidden_size_2)
        self.layer_4 = nn.Linear(hidden_size_2, num_outputs)

        # Initialise the rest of the weights
        self.init_weights()

        # Save parameters
        self.num_vars = num_vars
        self.lag = lag
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dp = dp

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # Forward propagation
    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: inputs with shape [batch size, lag * number of variables]
        :return: returns the forecast for the future value of the target variable.
        """
        aggregated = None

        # Propagate in sub-networks
        for i in range(self.num_vars):
            layer_1 = self.layer1_list[i]
            layer_2 = self.layer2_list[i]
            inp = inputs[:, (self.lag * i):(self.lag * (i + 1))]
            tmp = F.dropout(layer_2(F.dropout(layer_1(inp), p=self.dp, training=True)), p=self.dp, training=True)
            if i == 0:
                aggregated = self.imp_weights[i] * tmp
            else:
                aggregated = torch.cat((aggregated, self.imp_weights[i] * tmp), dim=1)

        # Final two layers
        pred = self.layer_4(F.dropout(self.layer_3(aggregated)), p=self.dp, training=True)

        return pred
