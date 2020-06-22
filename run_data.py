"""
A short script for getting acquainted with the data; converts data from MATLAB's .mat to .npy and .csv.
"""
import data_utils
import numpy as np
import matplotlib.pyplot as plt

time_series_all_patients = dict()
labels_all_patients = dict()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def load_patient(filepath, patient_index):

    # file handle
    f = data_utils.load(filepath)

    # get spectra and labels
    spectra = data_utils.get_spectra_metadata(f=f, patient_index=patient_index)
    raw_labels, raw_label_times = data_utils.get_label_metadata(f=f, patient_index=patient_index)

    # extract measurement and corresponding times
    measurements = spectra['time_series']
    measurement_times = spectra['measurement_times']
    mz = spectra['mz_vals']

    # focus on labeled measurements
    max_time = min(max(raw_label_times), max(measurement_times))
    min_time = max(min(raw_label_times), min(measurement_times))
    # cut measurements and times
    measurement_idx_to_keep = np.logical_and(measurement_times <= max_time, measurement_times >= min_time)
    raw_label_idx_to_keep = np.logical_and(raw_label_times <= max_time, raw_label_times >= min_time)
    measurements = measurements[measurement_idx_to_keep, :]
    measurement_times = measurement_times[measurement_idx_to_keep]
    # cut labels and times
    raw_labels = raw_labels[raw_label_idx_to_keep]
    raw_label_times = raw_label_times[raw_label_idx_to_keep]

    # label the measurements
    num_measurements = measurement_times.shape[0]
    labels = np.zeros(num_measurements)
    temporal_discrepancy = []
    for i in range(num_measurements):
        mtime = measurement_times[i]
        ltime, idx = find_nearest(raw_label_times, mtime)
        labels[i] = raw_labels[idx]
        temporal_discrepancy.append(np.abs(ltime - mtime))

    # some output
    temporal_discrepancy = np.array(temporal_discrepancy)
    print("Average temporal discrepancy: ", np.mean(temporal_discrepancy))
    print("STD  of temporal discrepancy: ", np.std(temporal_discrepancy))

    return measurements, labels, mz


def main():

    # Path to data
    filepath = "MSData/SLIMMBA_20190508/SLIMMBA_preprocessed_pos_20190508.mat"
    savepath = "MSData/SLIMMBA_20190508/"

    for i in range(13):

        # Select patient
        patient_index = i

        # Load selected patient
        print("Currently analyzing patient #", patient_index)
        print("--------------------------------")
        measurements, labels, mz = load_patient(filepath=filepath, patient_index=patient_index)
        print("The shape of measurement data is: [measurement index x MZ index] = ", measurements.shape)
        print("The shape of label data is: [measurement index] = ", labels.shape)
        print("The shape of mz data is: [mz index] = ", mz.shape)

        # Save the data accordingly
        np.save(savepath + "patient" + str(patient_index) + "_data.npy", measurements)
        np.save(savepath + "patient" + str(patient_index) + "_labels.npy", labels)
        np.savetxt(savepath + "patient" + str(patient_index) + "_data.csv", measurements, delimiter=",")
        np.savetxt(savepath + "patient" + str(patient_index) + "_labels.csv", labels, delimiter=",")

        # Save mz axis only once because it's the same for every patient
        if i == 0:
            np.save("mz.npy", mz)
            np.savetxt("mz.csv", mz, delimiter=",")


if __name__ == '__main__':
    main()
    print("Success")