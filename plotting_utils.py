"""
Some utility functions for plotting results.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_molecule_rem(measurements, labels, molecule_idx, subj_idx, mz):
    """
    Plots molecule count time series alongside with REM, non-REM sleep stages.

    :param measurements: 2D array with molecule count time series.
    :param labels: 1D array with sleep stage labels. REM sleep stage needs to be labelled by -1, non-REM stages need to
    be labelled by 0, -2, -4 or -5.
    :param molecule_idx: index of the molecule.
    :param subj_idx: index of the subject (time series replicate).
    :param mz: 1D array with molecule mass-to-charge ratios.
    :return: None
    """
    print("mz value is: ", mz[molecule_idx])
    print("index is: ", molecule_idx)
    fig, ax1 = plt.subplots()
    time_axis = np.arange(labels.shape[0])*10
    ax1.plot(time_axis, measurements[:, molecule_idx] - np.mean(measurements[:, molecule_idx]),
             marker=".", markersize=2, linewidth=1.0)
    ax1.set_xlabel('Time (sec)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("Zero-mean of molecule #"+str(molecule_idx)+" on "+str(mz[molecule_idx])+" m/z", color='blue')
    ax1.set_title("Subject #"+str(subj_idx))
    ax1.tick_params('y', color='blue')
    ymin, ymax = ax1.get_ylim()
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == 0), facecolor='red', alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -1), facecolor='green', alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -2), facecolor='red', alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -3), facecolor='red', alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -4), facecolor='red', alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -5), facecolor='red', alpha=0.5)

    fig.tight_layout()
    plt.show()


def plot_molecule_stages(measurements, labels, molecule_idx, subj_idx, mz):
    """
    Plots molecule count time series alongside with sleep stages.

    :param measurements: 2D array with molecule count time series.
    :param labels: 1D array with sleep stage labels. Awake is labelled by 0, REM is -1, S1 is -2, S2 is -3, S3 is -4 and
    S4 is -5.
    :param molecule_idx: index of the molecule.
    :param subj_idx: index of the subject (time series replicate).
    :param mz: 1D array with molecule mass-to-charge ratios.
    :return: None
    """
    print("mz value is: ", mz[molecule_idx])
    print("index is: ", molecule_idx)
    fig, ax1 = plt.subplots()
    time_axis = np.arange(labels.shape[0])*10
    ax1.plot(time_axis, measurements[:, molecule_idx] - np.mean(measurements[:, molecule_idx]),
             marker=".", markersize=2, linewidth=1.0)
    ax1.set_xlabel('Time (sec)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel("Zero-mean of molecule #"+str(molecule_idx)+" on "+str(mz[molecule_idx])+" m/z", color='blue')
    ax1.set_title("Subject #"+str(subj_idx))
    ax1.tick_params('y', color='blue')
    ymin, ymax = ax1.get_ylim()
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == 0), facecolor="#FF0000", alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -1), facecolor="green", alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -2), facecolor="#FF006B", alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -3), facecolor="#F100FF", alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -4), facecolor="#5000FF", alpha=0.5)
    ax1.fill_between(time_axis, ymin, ymax, where=(labels == -5), facecolor="#00FFD6", alpha=0.5)

    fig.tight_layout()
    plt.show()


def plot_molecule(measurements, labels, molecule_idx, mz, coloured_labels, colours):
    """
    Plots molecule count time series alongside with specified time point labels (e.g. sleep stages).

    :param measurements: 2D array with molecule count time series.
    :param labels: 1D array with time point labels.
    :param molecule_idx: index of the molecule.
    :param mz: 1D array with molecule mass-to-charge ratios.
    :param coloured_labels: array of time point labels to be highlighted in the plot.
    :param colours: array of colours that correspond to the labels specified in coloured_labels.
    :return: None
    """
    print("mz value is: ", mz[molecule_idx])
    print("index is: ", molecule_idx)
    fig, ax1 = plt.subplots()
    time_axis = np.arange(labels.shape[0])*10
    ax1.plot(time_axis, measurements[:, molecule_idx] - np.mean(measurements[:, molecule_idx]),
             marker=".", markersize=2, linewidth=1.0)
    ax1.set_xlabel(r'\textbf{Time, s}')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(r'\textbf{Zero-mean Intensity of Ion on '+str(np.round(mz[molecule_idx], 3))+' m/z}')
    ymin, ymax = ax1.get_ylim()
    cnt = 0
    for l in coloured_labels:
        ax1.fill_between(time_axis, ymin, ymax, where=(labels == l), facecolor=colours[cnt], alpha=0.5)
        cnt += 1
    fig.tight_layout()
    plt.show()


def plot_CI(lower, upper, xlab, ylab):
    for i in range(lower.shape[0]):
        plt.plot(np.array([i, i]), np.array([lower[i], upper[i]]))
        plt.plot(np.array([i - 0.1, i + 0.1]), np.array([lower[i], lower[i]]))
        plt.plot(np.array([i - 0.1, i + 0.1]), np.array([upper[i], upper[i]]))
    plt.axhline(y=0, color="r")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
