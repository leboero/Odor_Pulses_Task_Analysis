import numpy as np
import scipy
from scipy import signal
from scipy.stats import norm,entropy,linregress
from scipy.optimize import minimize, curve_fit
from scipy.io import savemat
import matplotlib.pyplot as plt

def get_pulses(choices,cum_odor,min_pulse,max_pulse):
    all_Nps = np.ceil(cum_odor).astype(int)
    ind_selected = np.logical_and(all_Nps>=min_pulse,all_Nps<=max_pulse)
    return choices[ind_selected], all_Nps[ind_selected]

def get_odor_profile_actual(odor_profile):
    kernel = np.load('odor_kernel_50ms.npy')
    pulse_onsets = (np.diff(odor_profile)>0).astype(float)
    pulse_onsets_shifted = np.concatenate((np.zeros((70,)),pulse_onsets,np.zeros(131)))
    odor_profile_actual = np.convolve(pulse_onsets_shifted,kernel,mode='same')
    return odor_profile_actual

def get_conv_odor(session):
    num_trials = session['num_trials']
    conv_odor = np.zeros((num_trials,5000))
    conv_weights_all = np.zeros((num_trials,5000))
    sniff_kernel = np.load('inhalation_kernel_fine_weights_active.npy')
    sniff_kernel = sniff_kernel/sniff_kernel.mean()
    for i_trial in range(num_trials):
        sniff_raw = np.append(session['trial_pre_breath'][i_trial],session['trial_breath'][i_trial])
        sniff = butter_lowpass_filter(sniff_raw,10,1000,3)
        sniff = (sniff - sniff.mean() +1)/2
        sniff_onset,_ = scipy.signal.find_peaks(sniff,distance=120)
        sniff_onset = sniff_onset[sniff_onset>2250]
        sniff_onset = sniff_onset[sniff_onset<7750]
        sniff_markers = np.zeros((10000,))
        conv_weights = np.zeros((10000,))
        for i in range(len(sniff_onset)-1):
            nsample = sniff_onset[i+1]-sniff_onset[i]
            conv_weights[sniff_onset[i]:sniff_onset[i+1]] = scipy.signal.resample(sniff_kernel,nsample)*250/nsample
        sniff_markers[sniff_onset] = 1
        sniff_sampling_epoch = sniff[2500:7500]
        conv_weights_sampling_epoch = conv_weights[2500:7500]
        odor_command = session['trial_odor'][i_trial]
        odor = get_odor_profile_actual(odor_command)[0:5000]
        odor_effective = odor*conv_weights_sampling_epoch
        conv_odor[i_trial,:] = odor_effective
        conv_weights_all[i_trial,:] = conv_weights_sampling_epoch
    return conv_odor, conv_weights_all

def get_sniff_histogram(session,shuffled=False):
    bins = np.linspace(0,250,16) #16
    num_trials = session['num_trials']
    sniff_hist = np.zeros((num_trials,15)) #15
        
    for i_trial in range(num_trials):
        sniff_raw = np.append(session['trial_pre_breath'][i_trial],session['trial_breath'][i_trial])
        if len(sniff_raw) == 11500:
            sniff_raw = sniff_raw[2500:]
            
        #print(len(sniff_raw))
        sniff = butter_lowpass_filter(sniff_raw,8,1000,3)
        sniff = (sniff - sniff.mean() +1)/2
        sniff_onset,_ = scipy.signal.find_peaks(sniff,distance=100)
        sniff_onset = sniff_onset[sniff_onset>2250]
        sniff_onset = sniff_onset[sniff_onset<7750]
        sniff_phase = np.zeros((10000,))

        for i in range(len(sniff_onset)-1):
            nsample = sniff_onset[i+1]-sniff_onset[i]
            sniff_phase[sniff_onset[i]:sniff_onset[i+1]] = scipy.signal.resample(np.arange(0,250),nsample)
        
        sniff_phase_sampling_epoch = sniff_phase[2500:]
        odor_command = session['trial_odor'][i_trial]

        if shuffled:
            n_pulses = (np.diff(odor_command)==100).sum()
            valve_onset = np.random.randint(0,5000,(n_pulses,))
        else:
            valve_onset = np.argwhere(np.diff(odor_command)==100)[0:5000]
        odor_onset = valve_onset + 20
        odor_phase = sniff_phase_sampling_epoch[odor_onset].squeeze()
        hist,_ = np.histogram(odor_phase,bins)
        sniff_hist[i_trial,:] = hist
    return sniff_hist

def get_sniff_durations(session):
    num_trials = session['num_trials']
    sniff_durations = np.zeros((num_trials,50))
    for i_trial in range(num_trials):
        sniff_raw = np.append(session['trial_pre_breath'][i_trial],session['trial_breath'][i_trial])
        sniff = butter_lowpass_filter(sniff_raw,8,1000,3)
        sniff = (sniff - sniff.mean() +1)/2
        sniff_onset,_ = scipy.signal.find_peaks(-sniff,distance=100)
        sniff_onset = sniff_onset[sniff_onset>2250]
        sniff_onset = sniff_onset[sniff_onset<7750]
        sniff_markers = np.zeros((10000,))
        sniff_duration = np.diff(sniff_onset)
        sniff_durations[i_trial,:len(sniff_duration)] = sniff_duration
    return sniff_durations

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def get_autocorr(session,n):
    high_trials = session['high_trials']
    autocorr = list()
    for i in range(n):
        autocorr.append(np.corrcoef(high_trials[i+1:],high_trials[:-(i+1)])[0][1])
    return np.array(autocorr)

def find_odor_onset_new(trial_odor):
    odor_onsets_list = list()

    for i_trial in range(len(trial_odor)):
        onset_marks = (np.diff(trial_odor[i_trial])>0).astype(float)
        pulse_onsets = np.argwhere(onset_marks==1).flatten()
        odor_onsets_list.append(pulse_onsets.tolist())
    return odor_onsets_list

def get_blanks_duration(odor_onsets_array):
    coef_var = np.zeros(len(odor_onsets_array))
    blanks_duration =  []
    for i in range(len(odor_onsets_array)):
        trial_odor_onsets = odor_onsets_array[i]
        new_array = np.insert(trial_odor_onsets, 0, 0)
        new_array = np.insert(new_array, len(new_array), 5000)
        blanks_ = np.diff(new_array)
        blanks_duration.append(blanks_)
        coef_var[i] = blanks_.std()/blanks_.mean()
    return blanks_duration, coef_var

