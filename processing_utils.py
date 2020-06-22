"""
Some utility functions for data pre-processing.
"""
import numpy as np
import scipy.signal
import pandas as pd


def construct_lagged_dataset(x_list, y_list, lag, auto_regress=True):
    """
    Constructs a set of training instances from the list of predictor and response time series.

    :param x_list: list with replicates of predictor time series containing 2D numpy arrays, where columns correspond to
    variables and rows to time.
    :param y_list: list with replicates of response time series containing 1D numpy arrays.
    :param lag: order of considered regressive relationships, specifies the horizon in the past of predictors to be used
    to forecast the future of the response.
    :param auto_regress: boolean flag identifying if the autoregressive relationships need to be considered, i.e. if
    the past of the response series itself needs to be included into the feature vector.
    :return: returns a triple with numpy arrays for instance features, responses and replicate labels. The number of
    features is lag * (p + 1), if auto_regress is true, and lag * p, otherwise, where p is the number of predictor
    variables. The vector of replicate labels indicates for each instance from which replicate in x_list it was
    produced.
    """
    sz = 0
    for p in range(len(x_list)):
        tmp = y_list[p]
        t_len = tmp.shape[0]
        n = t_len - lag
        sz = sz + n
    tmp = x_list[0]
    if auto_regress:
        big_x = np.zeros((int(sz), lag * (tmp.shape[1] + 1)))
    else:
        big_x = np.zeros((int(sz), lag * tmp.shape[1]))
    big_y = np.zeros((int(sz), ))
    replicate_labels = np.zeros((int(sz), ))
    cnt = 0
    for p in range(len(x_list)):
        x = x_list[p]
        y = y_list[p]
        for i in range(x.shape[0] - lag):
            if auto_regress:
                feature_vals = np.zeros((1, lag * (x.shape[1] + 1)))
            else:
                feature_vals = np.zeros((1, lag * x.shape[1]))
            for j in range(x.shape[1]):
                feature_vals[0, (j * lag):((j + 1) * lag)] = x[i:(i + lag), j]
            if auto_regress:
                feature_vals[0, (x.shape[1] * lag):((x.shape[1] + 1)*lag)] = y[i:(i + lag)]
            big_x[cnt, :] = feature_vals
            big_y[cnt] = y[i + lag]
            replicate_labels[cnt] = p
            cnt = cnt + 1
    return big_x, big_y, replicate_labels


def log_transform(x, a=1):
    """
    Performs log-transformation of the data.

    :param x: numpy array with data.
    :param a: constant to be added, to shift the data points (assuming x contains non-positive numbers).
    :return: returns a numpy array with log-transformed data.
    """
    x_ = x
    x_ = np.log(x_ + a)
    return x_


def total_area_norm(x):
    """
    Performs total ion count (TIC) normalisation of mass-spectrometry data.

    :param x: numpy array with data, wherein rows correspond to data points and columns correspond to metabolites
    (variables).
    :return: returns numpy array with normalised mass spectra.
    """
    x_ = np.zeros(x.shape)
    for i in range(0, x_.shape[0]):
        # Exclude possible outliers from computing the total ion count
        total_area = np.sum(np.abs(x[i, x[i, :] <= np.quantile(a=x[i, :], q=0.95)]))
        if total_area > 0:
            x_[i, :] = x[i, :] / total_area
    return x_


def quantile_norm(x):
    """
    Performs quantile normalisation of mass-spectrometry data.

    :param x: numpy array with data, wherein rows correspond to data points and columns correspond to metabolites
    (variables).
    :return: returns numpy array with normalised mass spectra.
    """
    x_ = np.zeros(x.shape)
    indices = np.zeros(x.shape)
    sorted = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        indices[i, :] = np.argsort(x[i, :])
        sorted[i, :] = x[i, indices[i, :].astype(int)]
    mus = np.mean(sorted, axis=0)
    for i in range(0, x.shape[0]):
        x_[i, :] = mus[indices[i, :].astype(int)]
    return x_


def medfc_norm(x):
    """
    Performs median fold-change normalisation of mass-spectrometry data.

    :param x: numpy array with data, wherein rows correspond to data points and columns correspond to metabolites
    (variables).
    :return: returns numpy array with normalised mass spectra.
    """
    x_ = np.zeros(x.shape)
    median_sample = np.median(x, axis=0)
    median_sample[median_sample == 0] = 0.0001
    for i in range(x.shape[1]):
        vec = x[i, :]
        coeffs = vec / median_sample
        x_[i, :] = vec / (np.median(coeffs) + 0.0001)
    return x_


def standardize(x):
    """
    Performs standardisation of the given multivariate time series.

    :param x: multivariate time series as a 2D numpy array, wherein columns correspond to variables.
    :return: returns standardised time series as a 2D numpy array, wherein each column is zero-mean and unit variance.
    """
    x_ = np.zeros(x.shape)
    for j in range(x.shape[1]):
        if np.std(x[:, j]) == 0:
            x_[:, j] = np.zeros(x.shape[0])
        else:
            x_[:, j] = (x[:, j] - np.mean(x[:, j])) / np.std(x[:, j])
    return x_


def smooth_spec_sgolay(x, w):
    """
    Performs mass spectrum smoothing using Savitzky-Golay filtering.

    :param x: 2D numpy array with MS data, columns corresponding to metabolites.
    :param w: the width of Savitzky-Golay filter window (consult the documentation of scipy.signal.savgol_filter for
    more details).
    :return: returns 2D numpy array with smoothed spectra.
    """
    x_ = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        x_[i, :] = scipy.signal.savgol_filter(x=x[i, :], window_length=w, polyorder=2)
        # Clip any counts that have become < 0 after smoothing
        x_[i, x_[i, :] <= 0] = 0
    return x_


def smooth_signal(x, w):
    """
    Performs time series smoothing using Savitzky-Golay filtering.

    :param x: 1D numpy array with univariate time series.
    :param w: the width of Savitzky-Golay filter window (consult the documentation of scipy.signal.savgol_filter for
    more details).
    :return: returns 1D numpy array with smoothed time series.
    """
    x_ = scipy.signal.savgol_filter(x=x, window_length=w, polyorder=2)
    return x_


def detrend(x, w):
    """
    Removes the trend component from the given multivariate time series.

    :param x: 2D numpy array with multivariate time series, wherein columns correspond to variables.
    :param w: the width of Savitzky-Golay filter window (consult the documentation of scipy.signal.savgol_filter for
    more details). This filter is used to estimate the trend curve, therefore, w should be sufficiently large to
    estimate a smooth trend.
    :return: returns 2D numpy array with detrended time series.
    """
    x_ = np.zeros(x.shape)
    for j in range(0, x_.shape[1]):
        trend = scipy.signal.savgol_filter(x=x[:, j], window_length=w, polyorder=4)
        x_[:, j] = x[:, j] - trend
    return x_


def sliding_variance(x, w):
    """
    Performs moving variance transformation of the univariate time series.

    :param x: time series as a 1D numpy array.
    :param w: window width for moving variance.
    :return: returns variance values for moving variance.
    """
    ts = pd.Series(x).cumsum()
    ts_ = ts.rolling(w, min_periods=0).std()**2
    x_ = ts_.values
    x_[0] = 0
    return ts_.values
