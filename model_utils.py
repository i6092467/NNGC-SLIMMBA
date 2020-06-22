"""
Various utility functions for constructing GC models and generating synthetic time series with known causal structures.
"""
import numpy as np
import mlp_gc as mlpgc
import lstm_gc as lstmgc
import processing_utils as utils
import torch.optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt


def get_causal_weights_mlp(x, y, categorical=False, lag=5, h_size_1=50, h_size_2=200, l_rate=0.0001, num_epochs=30,
                           lmbd=0.01, alpha=0.8, verbose=False, print_every=500):
    """
    Estimates Granger causality from predictor variables to the response using the approach based on MLP.

    :param x: predictor time series.
    :param y: response time series.
    :param categorical: flag identifying whether the response is categorically-valued.
    :param lag: model order.
    :param h_size_1: size of layers 1 and 2 in sub-networks.
    :param h_size_2: size of layer 3.
    :param l_rate: learning rate.
    :param num_epochs: number of training epochs.
    :param lmbd: regularisation parameter.
    :param alpha: trade off between L1 and L2 penalties.
    :param verbose: flag identifying whether print-outs are enabled.
    :param print_every: number of training steps between print-outs.
    :return: returns variable weights that can be used for identifying Granger causality.
    """
    num_vars = x[0].shape[1] + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Construct a lagged dataset for prediction which can be used for training
    big_x, big_y, rep_labels = utils.construct_lagged_dataset(x_list=x, y_list=y, lag=lag)
    # Set up the model and optimization method
    if not categorical:
        criterion = nn.MSELoss(reduction='mean')
        num_outputs = 1
    else:
        criterion = nn.CrossEntropyLoss()
        num_outputs = len(np.unique(np.concatenate(y)))
    model = mlpgc.MLPgc(num_vars=num_vars, device=device, lag=lag, hidden_size_1=h_size_1, hidden_size_2=h_size_2,
                        num_outputs=num_outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, amsgrad=True)
    started = False
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # Train for a number of epochs
    model.to(device)
    model.train()
    scheduler.step()
    for e in range(num_epochs):
        # Permute the data points, to ensure random order
        all_weights = np.zeros((big_x.shape[0], num_vars))
        inds = np.arange(0, big_x.shape[0])
        np.random.shuffle(inds)
        for i in range(big_x.shape[0]):
            if verbose and i != 0 and i % print_every == 0:
                if not started:
                    start = time.time()
                    started = True
                else:
                    end = time.time()
                    print(end-start)
                    started = True
                    start = time.time()
                print(weights)
                print((-weights).argsort()[:int(np.floor(0.1 * num_vars))])
                print(np.sort(weights))
                for j in range(num_vars):
                    plt.plot(all_weights[0:i, j], linewidth=0.7)
                plt.show()
                np.savetxt("weights.csv", weights)
            # One training point
            x_train = big_x[int(inds[i]), :]
            y_train = big_y[int(inds[i])]

            # Trasform for PyTorch
            inputs = Variable(torch.tensor(x_train, dtype=torch.float)).float().to(device)
            if not categorical:
                targets = Variable(torch.tensor(y_train, dtype=torch.float)).float().to(device)
            else:
                targets = Variable(torch.tensor(y_train, dtype=torch.long)).long().to(device)

            # Get the outputs from forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss
            if categorical:
                outputs = outputs.unsqueeze(0)
                targets = targets.unsqueeze(0)
            base_loss = criterion(outputs, targets)
            if categorical:
                base_loss = base_loss * big_y.shape[0] / np.sum(big_y == y_train)
            # Regularize the model
            l1_regularization = torch.norm(model.imp_weights, 1)
            l2_regularization = torch.norm(model.imp_weights, 2)
            loss = base_loss + lmbd * (alpha * l1_regularization + (1 - alpha) * l2_regularization)

            # Retreive causal influence weights
            weights = np.abs(model.imp_weights.data.numpy())
            all_weights[i, :] = weights

            # Make an optimization step
            loss.backward()
            optimizer.step()
    return weights


def get_causal_weights_lstm(x, y, categorical=False, lag_max=10, hidden_size_lstm=20, hidden_size_mlp=50, lmbd=0.01,
                            alpha=0.8, l_rate=0.0001, num_epochs=30, verbose=False, print_every=500):
    """
    Estimates Granger causality from predictor variables to the response using the approach based on LSTM.

    :param x: predictor time series.
    :param y: response time series.
    :param categorical: flag identifying whether the response is categorically-valued.
    :param lag_max: maximum lag of autoregressive relationships.
    :param hidden_size_lstm: size of the hidden state in the LSTM.
    :param hidden_size_mlp: size of the hidden layer in the MLP.
    :param lmbd: regularisation parameter.
    :param alpha: trade off between L1 and L2 penalties.
    :param l_rate: learning rate.
    :param num_epochs: number of training epochs.
    :param verbose: flag identifying whether print-outs are enabled.
    :param print_every: number of training steps between print-outs.
    :return: returns variable weights that can be used for identifying Granger causality.
    """
    num_vars = x[0].shape[1] + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up the model and optimization method
    if not categorical:
        criterion = nn.MSELoss(reduction='mean')
        num_outputs = 1
    else:
        criterion = nn.CrossEntropyLoss()
        num_outputs = len(np.unique(np.concatenate(y)))
    model = lstmgc.LSTMgc(num_vars=num_vars, device=device, lag_max=lag_max, hidden_size_lstm=hidden_size_lstm,
                          hidden_size_mlp=hidden_size_mlp, num_outputs=num_outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, amsgrad=True)
    started = False
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # Train for a number of epochs
    model.to(device)
    model.train()
    scheduler.step()
    for e in range(num_epochs):
        # Permute the data points, to ensure random order
        inds = np.arange(0, len(x))
        np.random.shuffle(inds)
        for i in range(len(x)):
            inds_t = np.arange(lag_max, x[int(inds[i])].shape[0] - 1)
            np.random.shuffle(inds_t)
            for t in range(lag_max, x[int(inds[i])].shape[0] - 1):
                t_ = inds_t[int(t - lag_max)]
                if verbose and t != 0 and t % print_every == 0:
                    if not started:
                        start = time.time()
                        started = True
                    else:
                        end = time.time()
                        print(end-start)
                        started = True
                        start = time.time()
                    print(weights)
                    print((-weights).argsort()[:5])
                # One training point
                x_train = np.concatenate([x[int(inds[i])][(t_-lag_max):t_, :],
                                          np.expand_dims(y[int(inds[i])][(t_-lag_max):t_], axis=1)], axis=1)
                y_train = y[int(inds[i])][t_]

                # Trasform for PyTorch
                inputs = Variable(torch.tensor(np.transpose(x_train), dtype=torch.float))
                inputs = inputs.unsqueeze(0)
                inputs = inputs.unsqueeze(3)
                targets = Variable(torch.tensor(y_train, dtype=torch.float))

                # Get the outputs from forward pass
                optimizer.zero_grad()
                outputs = model(inputs)

                # Loss
                if categorical:
                    outputs = outputs.unsqueeze(0)
                    targets = targets.unsqueeze(0)
                base_loss = criterion(outputs, targets)
                # Regularize the model
                l1_regularization = torch.norm(model.imp_weights, 1)
                l2_regularization = torch.norm(model.imp_weights, 2)
                loss = base_loss + lmbd * (alpha * l1_regularization + (1 - alpha) * l2_regularization)

                # Retrieve causal influence weights
                weights = np.abs(model.imp_weights.data.numpy())

                # Make an optimization step
                loss.backward()
                optimizer.step()
    return weights


def generate_timino_example_1(n, t):
    """
    Generates artificial dataset with time series X, Y and W. The causal structure is given by X -> Y and X -> W. All
    structural equations are linear with fixed coefficients. This example is taken from article "Causal Inference on
    Time Series using Structural Equation Models" by J. Peters, D. Janzing and B. Schölkopf.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.normal(0, 1, (t, ))**3
        eps_y = np.random.normal(0, 1, (t,))**3
        eps_w = np.random.normal(0, 1, (t,))**3
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        w = np.zeros((t, 1))
        for j in range(2, t):
            x[j, 0] = 0.3 * x[j - 1, 0] + 0.5 * eps_x[j]
            y[j, 0] = 0.8 * y[j - 1, 0] + 0.8 * x[j - 1, 0] + 0.5 * eps_y[j]
            w[j, 0] = -0.6 * w[j - 1, 0] + 0.8 * x[j - 2, 0] + 0.5 * eps_w[j]
        x_list.append(np.concatenate((x, y, w), axis=1))
    return x_list


def generate_timino_example_2(n, t):
    """
    Generates artificial dataset with time series X, W, Y and Z. The causal structure is given by X -> W -> Y and Y -> Z
    and W -> Z. All structural equations are linear with coefficients drawn from a uniform distribution for each
    replicate. There are instantaneous effects. This example is taken from article "Causal Inference on Time Series
    using Structural Equation Models" by J. Peters, D. Janzing and B. Schölkopf.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []
    for i in range(n):
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1, 1)
            if u_1 <= 0.5:
                a[k] = np.random.uniform(-0.8, -0.2, 1)
            else:
                a[k] = np.random.uniform(0.2, 0.8, 1)
        eps_x = 0.4 * np.random.normal(0, 1, (t, ))
        eps_y = 0.4 * np.random.normal(0, 1, (t,))
        eps_w = 0.4 * np.random.normal(0, 1, (t,))
        eps_z = 0.4 * np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        w = np.zeros((t, 1))
        z = np.zeros((t, 1))
        for j in range(1, t):
            x[j, 0] = a[0] * x[j - 1, 0] + eps_x[j]
            w[j, 0] = a[1] * w[j - 1, 0] + a[2] * x[j, 0] + eps_w[j]
            y[j, 0] = a[3] * y[j - 1, 0] + a[4] * w[j - 1, 0] + eps_y[j]
            z[j, 0] = a[5] * z[j - 1, 0] + a[6] * w[j, 0] + a[7] * y[j - 1, 0] + eps_z[j]
        x_list.append(np.concatenate((x, w, y, z), axis=1))
    return x_list


def generate_timino_example_3(n, t):
    """
    Generates artificial dataset with time series X, Y and Z. The causal structure is given by X -> Y -> Z. Structural
    equations feature non-linear reltaionships between variables. Moreover, additive noise terms are uniform, rather
    than Gaussian. This example is taken from article "Causal Inference on Time Series using Structural Equation Models"
    by J. Peters, D. Janzing and B. Schölkopf.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.uniform(-0.5, 0.5, (t, ))
        eps_y = np.random.uniform(-0.5, 0.5, (t,))
        eps_z = np.random.uniform(-0.5, 0.5, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        z = np.zeros((t, 1))
        for j in range(2, t):
            x[j, 0] = 0.8 * x[j - 1, 0] + 0.3 * eps_x[j]
            y[j, 0] = 0.4 * y[j - 1, 0] + (x[j - 1, 0] - 1)**2 + 0.3 * eps_y[j]
            z[j, 0] = 0.4 * z[j - 1, 0] + 0.5 * np.cos(y[j - 1, 0]) + np.sin(y[j - 1, 0]) + 0.3 * eps_z[j]
        x_list.append(np.concatenate((x, y, z), axis=1))
    return x_list


def generate_timino_example_4(n, t):
    """
    Generates artificial dataset with time series X and Y. The causal structure is given by X -> Y. The structural
    equation for Y is non-linear in past values of X (including a non-additive interaction between two lagged values).
    This example is taken from article "Causal Inference on Time Series using Structural Equation Models" by J. Peters,
    D. Janzing and B. Schölkopf.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.normal(0, 1, (t, ))
        eps_y = np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        for j in range(2, t):
            x[j, 0] = 0.2 * x[j - 1, 0] + 0.9 * eps_x[j]
            y[j, 0] = -0.5 + np.exp(-(x[j - 1, 0] + x[j - 2, 0])**2) + 0.1 * eps_y[j]
        x_list.append(np.concatenate((x, y), axis=1))
    return x_list


def generate_timino_example_5(n, t):
    """
    Generates artificial dataset with time series X and Y. The causal structure is given by X -> Y. The structural
    equation for Y is non-linear in the past value of X. This example is taken from article "Causal Inference on Time
    Series using Structural Equation Models" by J. Peters, D. Janzing and B. Schölkopf.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []
    for i in range(n):
        eps_x = 0.4 * np.random.normal(0, 1, (t, ))
        eps_y = 0.4 * np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        for j in range(2, t):
            x[j, 0] = -0.5 * x[j - 1, 0] + eps_x[j]
            y[j, 0] = -0.5 * y[j - 1, 0] + x[j - 1, 0]**2 + eps_y[j]
        x_list.append(np.concatenate((x, y), axis=1))
    return x_list


def generate_cat_ts(num_switch, num_states, t):
    """
    Generates one replicate of categorically-valued time series (transitions between states are uniform).

    :param num_switch: number of state switches in the replicate.
    :param num_states: number of states.
    :param t: length of time series.
    :return: returns 1D numpy array containing the time series.
    """
    switches = np.random.uniform(0, t - 1, num_switch)
    switches = np.sort(switches)
    switches = np.round(switches)
    switches = switches.astype(np.int)
    y = np.zeros((t,))
    for i in range(0, len(switches)):
        swi = switches[i]
        st = np.random.choice(a=range(0, num_states), size=1, )
        if i == 0:
            y[0:swi] = np.ones((swi,)) * st
        else:
            y[switches[i - 1]:swi] = np.ones((swi - switches[i - 1],)) * st
    return y


def generate_example_6(n, t):
    """
    Generates artifical dataset with continuously-valued time series X and binary-valued Y. The causal structure is
    Y -> X.

    :param n: number of time series replicates.
    :param t: length of time series.
    :return: returns the list of time series replicates.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = generate_cat_ts(int(np.floor(t / 50)), 2, t)
        y = y.reshape((t, 1))
        for j in range(1, t):
            x[j, 0] = -0.25 * x[j - 1, 0] + 0.35 * y[j - 1, 0] + 0.4 * eps_x[j]
        x_list.append(np.concatenate((x, y), axis=1))
    return x_list


def generate_example_7(n, t):
    """
    Generates time series X, W, Y, where X and W are continuously-valued and Y is binary-valued. The causal structure
    is given by X -> Y and X -> W <- Y. All dependencies are linear.

    :param n: number of time series replicates.
    :param t: length of time series.
    :return: returns the list of time series replicates.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.normal(0, 1, (t,))
        eps_w = np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        w = np.zeros((t, 1))
        y = np.zeros((t, 1))
        for j in range(5, t):
            x[j, 0] = 0.3 * x[j - 1, 0] + eps_x[j]
            w[j, 0] = -0.6 * w[j - 1, 0] + 0.25 * x[j - 1, 0] - 0.5 * y[j - 1, 0] + 0.3 * eps_w[j]
            if np.mean(x[(j-5):j, 0]) >= -0.25:
                y[j, 0] = 1
        x_list.append(np.concatenate((x, w, y), axis=1))
    return x_list


def generate_example_7_1(n, t):
    """
    Generates time series X, W, Y, where X and W are continuously-valued and Y is binary-valued. The causal structure
    is given by X -> Y and X -> W <- Y. Y depends on X non-linearly.

    :param n: number of time series replicates.
    :param t: length of time series.
    :return: returns the list of time series replicates.
    """
    x_list = []
    for i in range(n):
        eps_x = np.random.normal(0, 1, (t,))
        eps_w = np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        w = np.zeros((t, 1))
        y = np.zeros((t, 1))
        for j in range(5, t):
            x[j, 0] = 0.3 * x[j - 1, 0] + eps_x[j]
            w[j, 0] = -0.6 * w[j - 1, 0] + 0.25 * x[j - 1, 0] - 0.5 * y[j - 1, 0] + 0.3 * eps_w[j]
            if np.mean(x[(j-5):j, 0]) >= -0.25 and np.mean(x[(j-5):j, 0]) <= 0.25:
                y[j, 0] = 1
        x_list.append(np.concatenate((x, w, y), axis=1))
    return x_list


def generate_example_8(n, t):
    """
    Generates an artificial dataset with variables X, Y and Z. All dependencies are linear, autoregressive coefficients
    are drawn from a uniform distribution. The causal structure is given by X -> Y <- Z and X -> Z.

    :param n: number of replicates.
    :param t: length of time series.
    :return: returns list of time series replicates as 2D numpy arrays.
    """
    x_list = []

    u_xy = np.random.uniform(0, 1)
    a_xy = ((u_xy >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_xy < 0.5) * 1) * np.random.uniform(0.2, 0.8)
    u_zy = np.random.uniform(0, 1)
    a_zy = ((u_zy >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_zy < 0.5) * 1) * np.random.uniform(0.2, 0.8)
    u_xz = np.random.uniform(0, 1)
    a_xz = ((u_xz >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_xz < 0.5) * 1) * np.random.uniform(0.2, 0.8)

    u_xx = np.random.uniform(0, 1)
    a_xx = ((u_xx >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_xx < 0.5) * 1) * np.random.uniform(0.2, 0.8)
    u_yy = np.random.uniform(0, 1)
    a_yy = ((u_yy >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_yy < 0.5) * 1) * np.random.uniform(0.2, 0.8)
    u_zz = np.random.uniform(0, 1)
    a_zz = ((u_zz >= 0.5) * 1) * np.random.uniform(-0.8, -0.2) + ((u_zz < 0.5) * 1) * np.random.uniform(0.2, 0.8)

    for i in range(n):
        eps_x = np.random.normal(0, 1, (t,))
        eps_y = np.random.normal(0, 1, (t,))
        eps_z = np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        z = np.zeros((t, 1))
        for j in range(2, t):
            x[j, 0] = a_xx * x[j - 1, 0] + eps_x[j]
            y[j, 0] = a_yy * y[j - 1, 0] + a_xy * x[j - 2, 0] + a_zy * z[j - 1, 0] + eps_y[j]
            z[j, 0] = a_zz * z[j - 1, 0] + a_xz * x[j - 1, 0] + eps_z[j]
        x_list.append(np.concatenate((x, y, z), axis=1))
    return x_list


def generate_example_envir_1(t):
    """
    Generates artificial dataset with four replicates ('environments') of three variables X, Y, W. The causal structure
    is invariant across environments and is given by W <- X -> Y. In all environments the structural equations are
    linear, however, signs and magnitudes of coefficients differ.

    :param t: length of time series.
    :return: returns the list of time series replicates.
    """
    x_list = []

    # Environment 1
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += -0.25 * x[j - 2] + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 2
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.4 * x[j - 2] + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 3
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += -0.2 * x[j - 2] + 0.4 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 4
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.1 * x[j - 2] + 0.35 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    return x_list


def generate_example_envir_2(t):
    """
    Generates an artificial dataset with four replicates ('environments') of three variables X, Y, W. The causal
    structure is invariant across environments and is given by W <- X -> Y. However, the structural equations differ
    across environments.

    :param t: length of time series to be generated.
    :return: returns list with replicates of multivariate time series represented by 2D numpy arrays.
    """
    x_list = []

    # Environment 1
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.3 * (x[j - 2] - 1)**2 + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 2
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.6 * np.cos(x[j - 2]) + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 3
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.25 * (x[j - 2] - 1)**2 + 0.4 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    # Environment 4
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            if j > 1:
                w[j] += 0.3 * np.sin(x[j - 2]) + 0.35 * w[j - 1]
    x_list.append(np.concatenate((x, y, w), axis=1))

    return x_list


def generate_example_envir_3(t):
    """
    Generates an artificial dataset with three replicates ('environments') of four variables X, Y, Z, W. The causal
    structure is invariant across environments and is given by X -> Y, X -> Z, X -> W and Z -> W. However, the
    structural equations differ.

    :param t: length of time series to be generated.
    :return: returns list with replicates of multivariate time series represented by 2D numpy arrays.
    """
    x_list = []

    # Environment 1
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_z = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    z = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        z[j] += 0.3 * eps_z[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            z[j] += 0.2 * x[j - 1] + 0.3 * z[j]
            if j > 1:
                w[j] += 0.3 * x[j - 2] * z[j - 1] + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, z, w), axis=1))

    # Environment 2
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_z = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    z = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        z[j] += 0.3 * eps_z[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            z[j] += 0.2 * x[j - 1] + 0.3 * z[j]
            if j > 1:
                w[j] += 0.5 * x[j - 2] + 0.35 * z[j - 1] + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, z, w), axis=1))

    # Environment 3
    eps_x = np.random.normal(0, 1, (t,))
    eps_y = np.random.normal(0, 1, (t,))
    eps_z = np.random.normal(0, 1, (t,))
    eps_w = np.random.normal(0, 1, (t,))
    x = np.zeros((t, 1))
    y = np.zeros((t, 1))
    z = np.zeros((t, 1))
    w = np.zeros((t, 1))
    for j in range(t):
        x[j] += 0.3 * eps_x[j]
        y[j] += 0.3 * eps_y[j]
        z[j] += 0.3 * eps_z[j]
        w[j] += 0.3 * eps_w[j]
        if j > 0:
            x[j] += 0.6 * x[j - 1]
            y[j] += 0.3 * x[j - 1] + 0.5 * y[j - 1]
            z[j] += 0.2 * x[j - 1] + 0.3 * z[j]
            if j > 1:
                w[j] += np.cos(0.2 * x[j - 2] + 0.6 * z[j - 1]) + 0.5 * w[j - 1]
    x_list.append(np.concatenate((x, y, z, w), axis=1))

    return x_list
