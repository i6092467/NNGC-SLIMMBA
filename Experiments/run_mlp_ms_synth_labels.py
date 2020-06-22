"""
Runs the simulation experiment with an artificially generated REM signal.
"""
import numpy as np
import mlp_gc as mlpgc
import processing_utils
import torch.optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.metrics
import sys
import model_utils

data_dir = "MSData/SLIMMBA_20190508/positive/"    # Directory with data as .npy or .csv files
out_file = "weights_final.csv"

x_list = []             # List with metabolic time series of different subjects
y_list = []             # List with sleep stage time series of different subjects
y_list_one_hot = []     # List with one-hot encoding of sleep satges
n_patients = 13         # Number of subjects
n_molecules = 1271      # Number of molecules
n_sleep_stages = 2      # Number of sleep stages

lag = 30                # Lag for auto-regressive model
n_hidden_units_1 = 100  # Number of hidden units in sub-networks of the MLP
n_hidden_units_2 = 200  # Number of hidden units in the common layer of the MLP
lr = 0.0001             # Learning rate
n_epochs = 1            # Number of training epochs
lmb = 0.001              # Regularisation (lambda) parameter
alpha = 0.8             # Parameter controlling the tradeoff between L1 and L2 regularisation
mini_batch_size = 100   # Size of mini-batches

ignored_subjects = []  # List of subject indices to be ignored in the inference

B = 100


def load_data():
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("Loading data\n")
    for i in range(0, n_patients):
        currx = np.loadtxt(data_dir + "patient" + str(i) + "_data.csv")
        curry = np.loadtxt(data_dir + "patient" + str(i) + "_labels.csv")

        # NB: Replace original response with artificially generated categorical time series
        n_switches = np.random.choice([4, 6], 1)
        curry = model_utils.generate_cat_ts(n_switches, 2, len(curry))
        y_list.append(curry)
        x_list.append(currx)
    sys.stdout.write("---------------------------------------------\n")
    return


def pre_process():
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("Pre-processing data\n")
    for i in range(0, n_patients):
        sys.stdout.write("Pre-processing subject #"+str(i)+"\n")
        currx = x_list[i]
        currx = processing_utils.standardize(currx)   # Zero-mean, unit-variance
        x_list[i] = currx
    sys.stdout.write("---------------------------------------------\n")
    return


def construct_lagged_dataset(x_list, y_list, lag, auto_regress=True, ignored_subjects=[]):
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("Constructing training instances\n")
    # Compute the number of potential training points to allocate memory
    sz = 0
    for p in range(len(x_list)):
        tmp = y_list[p]
        t_len = tmp.shape[0]
        n = t_len - lag
        sz = sz + n

    # Do we include the response time series as a predictor?
    if auto_regress:
        big_x = np.zeros((int(sz), lag * (n_molecules + 1)))
    else:
        big_x = np.zeros((int(sz), lag * n_molecules))
    big_y = np.zeros((int(sz), ))

    # Go through subjects and construct training instances
    replicate_labels = np.zeros((int(sz), ))
    cnt = 0
    for p in range(len(x_list)):
        # Ignore some subjects (if needed)
        if p not in ignored_subjects:
            x = x_list[p]
            y = y_list[p]
            for i in range(x.shape[0] - lag):
                # Ignore instances which contain points with missing sleep stage labels
                if not((y[i + lag] == 6 or y[i + lag] == 7) or (np.any(y[i:(i + lag)] == 6) or
                                                                np.any(y[i:(i + lag)] == 7))):
                    if auto_regress:
                        feature_vals = np.zeros((1, lag * (x.shape[1] + 1)))
                    else:
                        feature_vals = np.zeros((1, lag * x.shape[1]))

                    for j in range(x.shape[1]):
                        feature_vals[0, (j * lag):((j + 1) * lag)] = x[i:(i + lag), j]

                    if auto_regress:
                        feature_vals[0, (x.shape[1] * lag):((x.shape[1] + 1) * lag)] = y[i:(i + lag)]
                    big_x[cnt, :] = feature_vals
                    big_y[cnt] = y[i + lag]
                    replicate_labels[cnt] = p
                    cnt = cnt + 1

    # Remove unused parts of arrays
    if cnt < big_x.shape[0] - 1:
        big_x = big_x[0:(cnt - 1), :]
        big_y = big_y[0:(cnt - 1)]
        replicate_labels = replicate_labels[0:(cnt - 1)]

    sys.stdout.write("We have " + str(big_x.shape[0]) + " training instances\n")
    sys.stdout.write("---------------------------------------------\n")

    return big_x, big_y, replicate_labels


def get_causal_weights_mlp(x_list, y_list, lag, h_size_1, h_size_2, l_rate, num_epochs, lmbd, alpha=0.8,
                           ignored_subjects=[], use_cuda=False, weighted_classes=True, verbose=False, plotting=False,
                           print_every=1000):
    # Number of predictors
    num_vars = n_molecules + 1

    # Use specified device (CPU or GPU)
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        sys.stdout.write("---------------------------------------------\n")
        sys.stdout.write("Using " + str(device) + "\n")
        sys.stdout.write("---------------------------------------------\n")

    # Construct a dataset with lagged instances
    big_x, big_y, rep_labels = construct_lagged_dataset(x_list, y_list, lag=lag,
                                                        ignored_subjects=ignored_subjects)
    class_weights = np.array([0.1, 0.9])
    sys.stdout.write("---------------------------------------------\n")
    sys.stdout.write("Class weights are " + str(class_weights) + "\n")
    sys.stdout.write("---------------------------------------------\n")
    class_weights = torch.Tensor(class_weights).float().to(device)      # Transform to tensor for PyTorch

    # Set up the model and optimization method
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    num_outputs = n_sleep_stages

    model = mlpgc.MLPgc(num_vars=num_vars, device=device, lag=lag, hidden_size_1=h_size_1, hidden_size_2=h_size_2,
                        num_outputs=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

    # Training loop
    started = False
    model.to(device)
    model.train()
    weights = np.abs(model.imp_weights.data.cpu().numpy())

    for e in range(num_epochs):
        if verbose:
            sys.stdout.write("#############################################\n")
            sys.stdout.write("Epoch " + str(e) + "\n")

        # Permute to ensure a random ordering of training instances
        inds = np.arange(0, big_x.shape[0])
        np.random.shuffle(inds)

        incurred_loss = 0   # Keep track of incurred loss periodically

        batch_cnt = 0

        for i in range(0, big_x.shape[0], mini_batch_size):

            # Print and save current weights
            if verbose and batch_cnt != 0 and batch_cnt % print_every == 0:
                sys.stdout.write("---------------------------------------------\n")
                if not started:
                    start = time.time()
                    started = True
                else:
                    end = time.time()
                    sys.stdout.write("Elapsed time: " + str(end-start) + "\n")
                    started = True
                    start = time.time()
                # Print molecule indices with highest weights
                sys.stdout.write("Used " + str(i) + " instances so far\n")
                sys.stdout.write("Top molecules: " + str((-weights).argsort()[:20]) + "\n")
                sys.stdout.write("Top weights: " + str(weights[(-weights).argsort()[:20]]) + "\n")
                sys.stdout.write("Loss incurred: " + str(incurred_loss) + "\n")
                sys.stdout.write("Confusion matrix (for the last batch): \n")
                cm = sklearn.metrics.confusion_matrix(y_true=targets.data.cpu(),
                                                      y_pred=np.argmax(outputs.data.cpu(), 1))
                sys.stdout.write(str(cm))
                sys.stdout.write("\n")
                print("---------------------------------------------\n")
                incurred_loss = 0

            # Retrieve training instances for this batch
            x_train = big_x[inds[i:(i + mini_batch_size)].astype(int), :]
            y_train = big_y[inds[i:(i + mini_batch_size)].astype(int)]

            # Transform to tensors for PyTorch
            inputs = Variable(torch.tensor(x_train, dtype=torch.float)).float().to(device)
            targets = Variable(torch.tensor(y_train, dtype=torch.long)).long().to(device)

            # Get the outputs from forward pass
            outputs = model(inputs)

            # Loss
            base_loss = criterion(outputs, targets)

            # Regularize the model
            l1_regularization = torch.norm(model.imp_weights, 1)
            l2_regularization = torch.norm(model.imp_weights, 2)
            loss = base_loss + lmbd * (alpha * l1_regularization + (1 - alpha) * l2_regularization)

            # Incur loss
            incurred_loss += loss.data.cpu().numpy()

            # Retrieve causal influence weights
            weights = np.abs(model.imp_weights.data.cpu().numpy())

            # Make an optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_cnt += 1
    return model.imp_weights.data.cpu().numpy()


def bootstrap_weights(b):
    weights = np.zeros((b, n_molecules + 1))
    subject_inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    for i in range(b):
        sys.stdout.write("---------------------------------------------\n")
        sys.stdout.write("Fitting bootstrapped model " + str(i) + "\n")
        # Sample subjects with replacement
        resampled_subj = np.random.choice(a=subject_inds, size=(subject_inds.shape[0], ), replace=True)
        sys.stdout.write("Sampled subjects are " + str(resampled_subj) + "\n")
        # Construct resampled dataset
        x_list_i = []
        y_list_i = []
        for j in range(resampled_subj.shape[0]):
            x_list_i.append(x_list[int(resampled_subj[j])])
            y_list_i.append(y_list[int(resampled_subj[j])])
        w = get_causal_weights_mlp(x_list=x_list_i, y_list=y_list_i, lag=lag,
                                   h_size_1=n_hidden_units_1, h_size_2=n_hidden_units_2, l_rate=lr, num_epochs=n_epochs,
                                   lmbd=lmb, alpha=alpha, ignored_subjects=[], use_cuda=True, verbose=True,
                                   print_every=10, plotting=False)
        weights[i, ] = w
        sys.stdout.write("---------------------------------------------\n")
    return weights


def run_gc_analysis():
    """

    :return:
    """
    w = bootstrap_weights(b=B)
    np.savetxt(out_file, w)


def main():
    global out_file, lmb

    if len(sys.argv) >= 2:
        out_file = sys.argv[1]
    sys.stdout.write("Output file name: " + out_file + "\n")

    if len(sys.argv) >= 3:
        lmb = float(sys.argv[2])
    sys.stdout.write("Lambda: " + str(lmb) + "\n")

    load_data()
    pre_process()

    millis = int(round(time.time() * 1000)) % 10000000
    sys.stdout.write("\nSETTING SEED TO " + str(millis) + "\n")
    np.random.seed(millis)

    run_gc_analysis()


if __name__ == '__main__':
    main()
    sys.stdout.write("Success\n")
