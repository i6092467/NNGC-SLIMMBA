"""
Some utility methods for handling h5py files.
"""
import numpy as np
import h5py
import datetime as dt

# Some handy default values
SPECTRA_KEY = 'TT'
EEG_KEY = 'EEG'
TIMES_KEY = 'TIMES'


def load(filepath):
    """ 
    Loads the h5py file handle for the given file

    :param filepath: str
    :returns f, h5py.File handle
    """
    return h5py.File(filepath)


def datenum2datetime(matlab_datenum):
    """
    Converts matlab datenum format to datetime object

    :param matlab_datenum: datenum object from MATLAB
    :returns datetime object
    """

    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days=366)

    return day + dayfrac


def get_spectra_metadata(f, patient_index, verbose=False):
    """
    Gets metadata associated with spectra for the given patient

    :param f: the h5py File handle
    :param patient_index: the patient for which we need the spectra data
    :param verbose: Whether to print during loading
    :returns spectra_dict: A dictionary with attribute values

    """
    # h5py reference to spectra object
    spectra_ref = f.get(SPECTRA_KEY).value.T
    spectra = spectra_ref[patient_index]

    # The organization of spectra goes as
    # 0: filenames, 1: mz_vals, 2: time_series 3: measurement_times 4: clock_times

    # Holds the different spectra attributes 
    # We only use mz_vals, measurement_times, time_series
    spectra_dict = {}

    # Loads mz_vals
    name = h5py.h5r.get_name(spectra[1], f.id)
    mz_vals = np.array(f[name]).flatten()
    spectra_dict['mz_vals'] = mz_vals
    if verbose:
        print('Loaded mz_vals. Shape {}'.format(mz_vals.shape))

    # Loads time_series
    name = h5py.h5r.get_name(spectra[2], f.id)
    time_series = np.array(f[name])
    spectra_dict['time_series'] = time_series
    if verbose:
        print('Loaded time series. Shape {}'.format(time_series.shape))

    # Loads measurement times
    name = h5py.h5r.get_name(spectra[3], f.id)
    measurement_times = np.array(f[name]).flatten()
    spectra_dict['measurement_times'] = measurement_times
    if verbose:
        print('Loaded measurement times. Shape {}'.format(measurement_times.shape))

    return spectra_dict


def get_label_metadata(f, patient_index, verbose=False):
    """ Gets the labels and associated label add times for given patient
    
    :param f: h5py File handle
    :param patient_index: Which patient we want the data for
    :verbose bool, whether to display during loading

    """
    
    # times of shape 2 x 11, with 1st column, indicating attribute names and second the values
    # We need ms_start (in datenum), which is the first attribute
    times_ref = f.get(TIMES_KEY).value

    name = h5py.h5r.get_name(times_ref[1, 0], f.id)
    ms_starts = f[name]
    ms_start_patient = ms_starts[patient_index][0]

    # Convert the datenum to datetime
    ms_start_datetime = datenum2datetime(ms_start_patient)
    
    EEG_ref = f.get(EEG_KEY).value
    labels_ref = EEG_ref[1:, -2] #DO NOT CHANGE

    # Load labels and add times
    name = h5py.h5r.get_name(labels_ref[patient_index], f.id)
    label_data = f[name]

    labels = np.array(label_data[1, :]).flatten()
    label_times_datenum = label_data[0, :].flatten()
    label_times_datetime = list(map(datenum2datetime, label_times_datenum))
    label_times_seconds = np.array(list(map(lambda x: x.timestamp(), label_times_datetime)))
    label_times_seconds = label_times_seconds - ms_start_datetime.timestamp()

    if verbose:
        print('Loaded label data. Shape {}'.format(labels.shape))

    assert len(labels) == len(label_times_seconds)

    return labels, label_times_seconds
