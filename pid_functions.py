import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from statistics import median, mean, stdev, mode
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal import butter, filtfilt, iirfilter, sosfiltfilt, savgol_filter, iirnotch



import scipy.integrate as integrate
import scipy.special as special

def flatten(lst):
    """Flatten a nested list."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bp_bessel(signal, fs=1000, low_cut=0.2, high_cut=200, order=2):
    nyq = fs/2
    low = low_cut/nyq
    high = high_cut/nyq
    sos = iirfilter(order, [low, high], btype='band', ftype='bessel', output='sos')
    return sosfiltfilt(sos, signal)#, method='gust')

def bp_filter(signal):
    fs = 1000
    low_cut = 0.3
    high_cut = 200
    
    nyq = 0.5 * fs
    low = low_cut/nyq
    high = high_cut/nyq
    
    order = 2
    
    b,a = butter(order, [low, high], 'bandpass', analog=False)
    y = filtfilt(b, a, signal, axis=0)
    
    return(y)

def height_threshold(signal, begin, end, n_sigma):
    return np.median(signal[begin:end])+np.std(signal)*n_sigma

def peak_parameters(datax, threshold):
    peaks, _ = find_peaks(datax, height=threshold, distance=15, width=15)
    halfwidths = peak_widths(datax, peaks, rel_height=0.5)[0].tolist()
    fullwidth_markers = peak_widths(datax, peaks, rel_height=0.9)
    rise_times = (peaks-fullwidth_markers[2]).tolist()
    return halfwidths, rise_times, fullwidth_markers[0].tolist()

def peak_blank_analyzer(datax, threshold):  
    whiff_widths = []
    whiff_segments = []
    blank_indexes = []
    blank_durations = []

    peaks, _ = find_peaks(datax, height=threshold, distance=15, width=15)
    # Loop through each peak and find where the signal crosses the threshold
    for peak in peaks:
        # Find left intersection: where the signal crosses the threshold going up, before the peak
        left_intersections = np.where(np.array(datax[:peak]) < threshold)[0]
        left = left_intersections[-1] if len(left_intersections) > 0 else 0

        # Find right intersection: where the signal crosses the threshold going down, after the peak
        right_intersections = np.where(np.array(datax[peak:]) < threshold)[0]
        right = right_intersections[0] + peak if len(right_intersections) > 0 else len(datax) - 1

        # Check if the current segment overlaps with the last added one
        if len(whiff_segments) == 0 or (left != whiff_segments[-1][0] or right != whiff_segments[-1][1]):
            width = right - left
            whiff_segments.append((left, right))
            whiff_widths.append(width)

    for i in range(len(whiff_segments) - 1):
        right_of_current = whiff_segments[i][1]
        left_of_next = whiff_segments[i + 1][0]

        # Collect subthreshold data points
        if left_of_next > right_of_current:
            subthreshold_indices = list(range(right_of_current, left_of_next))
            blank_indexes.append(subthreshold_indices)
            blank_durations.append(len(subthreshold_indices))

    # Add subthreshold interval before the first peak if applicable
    if whiff_segments[0][0] > 0:
        subthreshold_indices = list(range(0, whiff_segments[0][0]))
        blank_indexes.append(subthreshold_indices)
        blank_durations.insert(0, len(subthreshold_indices))

    # Add subthreshold interval after the last peak if applicable
    if whiff_segments[-1][1] < len(datax) - 1:
        subthreshold_indices = list(range(whiff_segments[-1][1], len(datax)))
        blank_indexes.append(subthreshold_indices)
        blank_durations.append((len(subthreshold_indices)))


    # Output the unique widths
    #print("Unique Widths:", whiff_widths)
    #print("Subthreshold Intervals (indices and widths):", blank_durations)

    # Plotting for visualization
    fig = plt.figure(figsize=(10,8))
    plt.plot(datax, label='Data')
    plt.plot(peaks, np.array(datax)[peaks], "x", label='Peaks')
    plt.hlines(threshold, 0, len(datax) - 1, color="gray", linestyle="--", label=f'Height = {threshold}')
    for left, right in whiff_segments:
        plt.hlines(threshold, left, right, color="C2", label='Unique Width' if left == whiff_segments[0][0] else "")
    for indices in blank_indexes:
        plt.plot(indices, [threshold - 0.01] * len(indices), color='red', lw=2, label='Subthreshold' if indices == blank_indexes[0] else "")
    #fig.legend()
    
    return whiff_widths, blank_durations

def filter_signal(signal, fs=1000, method="bandpass", **kwargs):
    """
    Flexible filter toolbox for 1D signals.

    Parameters
    ----------
    signal : array
        Input signal.
    fs : int
        Sampling frequency (Hz).
    method : str
        Filtering method. Options:
            - "bandpass"
            - "highpass"
            - "lowpass"
            - "notch"
            - "bandstop"
            - "savgol"
            - "wavelet"
    kwargs : dict
        Extra arguments depending on the method:
            bandpass: low_cut, high_cut, order
            highpass: cutoff, order
            lowpass: cutoff, order
            notch: f0, Q
            bandstop: low, high, order
            savgol: window, poly
            wavelet: wavelet, level
    """

    method = method.lower()

    if method == "bandpass":
        low_cut = kwargs.get("low_cut", 0.5)
        high_cut = kwargs.get("high_cut", 200)
        order = kwargs.get("order", 2)
        nyq = fs / 2
        b, a = butter(order, [low_cut/nyq, high_cut/nyq], btype="band")
        return filtfilt(b, a, signal)

    elif method == "highpass":
        cutoff = kwargs.get("cutoff", 1.0)
        order = kwargs.get("order", 2)
        nyq = fs / 2
        b, a = butter(order, cutoff/nyq, btype="high")
        return filtfilt(b, a, signal)

    elif method == "lowpass":
        cutoff = kwargs.get("cutoff", 100)
        order = kwargs.get("order", 2)
        nyq = fs / 2
        b, a = butter(order, cutoff/nyq, btype="low")
        return filtfilt(b, a, signal)

    elif method == "notch":
        f0 = kwargs.get("f0", 60)   # notch frequency
        Q = kwargs.get("Q", 30.0)   # quality factor
        b, a = iirnotch(f0, Q, fs)
        return filtfilt(b, a, signal)

    elif method == "bandstop":
        low = kwargs.get("low", 48)
        high = kwargs.get("high", 52)
        order = kwargs.get("order", 2)
        nyq = fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype="bandstop")
        return filtfilt(b, a, signal)

    elif method == "savgol":
        window = kwargs.get("window", 101)  # must be odd
        poly = kwargs.get("poly", 3)
        return savgol_filter(signal, window_length=window, polyorder=poly)

    # elif method == "wavelet":
    #     wavelet = kwargs.get("wavelet", "db4")
    #     level = kwargs.get("level", 3)
    #     coeffs = pywt.wavedec(signal, wavelet, mode="symmetric")
    #     sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    #     uthresh = sigma * np.sqrt(2*np.log(len(signal)))
    #     coeffs[1:] = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    #     return pywt.waverec(coeffs, wavelet, mode="symmetric")

    else:
        raise ValueError(f"Unknown method '{method}'")