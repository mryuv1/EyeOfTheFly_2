import numpy as np


def unit_filter(signal):
    return np.array(signal)


# Receives a signal in Time Domain,  desired cutoff frequency, sampling rate, order is set as 5
# Returns the signal in Time Domain after filtering
def butter_lpf(signal, cutoff_frequency, fs, order=5):
    nyq = 0.5 * fs
    normalised_cutoff = cutoff_frequency / nyq
    # butter returns Numerator (b) and Denominator (a) polynomials of the IIR filter
    b, a = butter(order, normalised_cutoff, btype='low', analog=False)  # Butterworth filter design
    filtered_signal = lfilter(b, a, signal)  # Filter data along one-dimension with the designed IIR filter
    return filtered_signal


def delay_filter(signal, stride=1):
    signal = np.array(signal)
    length = signal.shape[0]
    return np.pad(signal[stride:], [0, stride], mode='constant')

