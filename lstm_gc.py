"""
Modules for inferring Granger causality based on long short-term memory networks (LSTMs).
"""
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class LSTMgc(nn.Module):
    # This class defines the LSTM model for inferring Granger causality in a multivariate time series
    def __init__(self, num_vars, device, lag_max, hidden_size_lstm, hidden_size_mlp, num_outputs=1):
        """
        Initialises an LSTMgc module, which represents a neural network model for Granger causality estimation.

        :param num_vars: number of variables, including the response.
        :param device: device to be used for calculations, CPU or GPU.
        :param lag_max: input size for nn.LSTMCell module.
        :param hidden_size_lstm: size of hidden states in LSTMs.
        :param hidden_size_mlp: size of hidden layer in MLP.
        :param num_outputs: number of output units.
        """
        super(LSTMgc, self).__init__()

        # LSTMs
        self.lstm_cell_list = nn.ModuleList()
        for state in range(num_vars):
            self.lstm_cell_list.append(nn.LSTMCell(lag_max, hidden_size_lstm))

        # MLP for prediction
        self.pred_mlp_l1 = nn.Linear(hidden_size_lstm * num_vars, hidden_size_mlp)
        self.pred_mlp_l2 = nn.Linear(hidden_size_mlp, num_outputs)

        # Initialise weights for each variable
        self.imp_weights = nn.Parameter(torch.Tensor(np.ones((num_vars,)) / num_vars + np.random.normal(0, 0.00001,
                                                                                                        (num_vars,))))

        # Initialise weights
        self.init_weights()

        # Save parameters
        self.num_vars = num_vars
        self.lag = lag_max
        self.hidden_size_lstm = hidden_size_lstm
        self.hidden_size_mlp = hidden_size_mlp

        # Initialise LSTM states
        self.lstm_state_list = []
        for state in range(num_vars):
            self.lstm_state_list.append((Variable(torch.zeros(1, self.hidden_size_lstm).float()).to(device),
                                         Variable(torch.zeros(1, self.hidden_size_lstm).float()).to(device)))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: Inputs with shape [batch size, number of variables, sequence length, variable dimension]
        :return: returns the forecast of the future value of the target variable.
        """

        # Concatenate LSTM hidden state vectors into one large vector,
        # which will be then used for prediction
        aggregated = []
        cnt = 0
        for state, (lstm_cell, lstm_state) in enumerate(zip(self.lstm_cell_list, self.lstm_state_list)):
            lstm_state = lstm_cell(inputs[:, state, :, :].view(inputs.shape[0], -1), lstm_state)
            aggregated.append(lstm_state[1] * self.imp_weights[cnt])
            cnt += 1
        aggregated = torch.cat(aggregated, dim=1)

        # Calculate predictions
        pred = F.relu(self.pred_mlp_l1(aggregated))
        pred = self.pred_mlp_l2(pred)

        return pred
