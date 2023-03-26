# -*- coding: utf-8 -*-
# """
# Created on Sat Jun 18 15:37:00 2022
# @author: Dong HUANG
# Contact: huang_dong@tju.edu.cn
# """

# %% Import
import functools
import os
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import butter, filtfilt


def butter_notchstop(notch,Q,fs):
    b, a = signal.iirnotch(notch, Q,fs)
    return b, a


def preprocess_norm(eeg_data):
    scale_mean = np.mean(eeg_data, axis=-1, keepdims=True)
    scale_std = np.std(eeg_data, axis=-1, keepdims=True)
    eeg_data = (eeg_data - scale_mean) / (scale_std + 1e-8)

    return eeg_data


def preprocess_norm_layer(eeg_data):
    scale_mean = np.mean(eeg_data, axis=1, keepdims=True)
    scale_std = np.std(eeg_data, axis=1, keepdims=True)
    eeg_data = (eeg_data - scale_mean) / (scale_std + 1e-5)

    return eeg_data


@functools.lru_cache()
def cache_read(filename: str):
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        file_content = pd.read_csv(filename)
    else:
        raise NotImplementedError

    return file_content


def preprocess_ref(eeg_data, type='average', dataset='THU-EP'):
    if type == 'average':
        return eeg_data - np.mean(eeg_data)
    elif type in ['bipolar_std', 'mastoid_std']:
        reference_loopup_csv_name = './mid_files/reference_THU_EP.csv'
        remove_last_2 = True

        rlc_df = cache_read(reference_loopup_csv_name)
        if type == 'bipolar_std':
            index1, index2 = rlc_df['index1'] - 1, rlc_df['index2'] - 1
            if remove_last_2:
                index1, index2 = index1[:-2], index2[:-2]
            return eeg_data[index1, :] - eeg_data[index2, :]
        elif type == 'mastoid_std':
            index1, index2 = rlc_df['index1'] - 1, rlc_df['index3'] - 1
            if remove_last_2:
                index1, index2 = index1[:-2], index2[:-2]
            return eeg_data[index1, :] - eeg_data[index2, :]
    elif type == 'none':
        return eeg_data
    else:
        raise NotImplementedError


def preprocess_filt(data, low_cut=0.5, high_cut=45, fs=250, order=6):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    proced = signal.filtfilt(b, a, data)
    return proced


def preprocess_bsfilt(data, low_cut=48, high_cut=52, fs=250):
    # win = firwin(11, [low_cut, high_cut], pass_zero='bandpass', fs=fs)
    # proced = lfilter(win, 1, data)
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(6, [low, high], btype='bandstop')
    proced = signal.filtfilt(b, a, data)
    return proced


def preprocess_notch(data, notch=50, Q=35, fs=250):
    # notch, Q, fs = 50, 35, 200
    b, a = butter_notchstop(notch, Q, fs)
    filted_eeg_rawdata = filtfilt(b, a, data)
    return filted_eeg_rawdata


def preprocess_hpfilt(data, low_cut=0.1, fs=250):
    b, a = butter(11, low_cut, btype='hp', fs=fs)
    filted_data = filtfilt(b, a, data)
    return filted_data


def preprocess_01norm(data):
    scale_min = np.min(data, axis=-1, keepdims=True)
    scale_max = np.max(data, axis=-1, keepdims=True)
    eeg_data = (data - scale_min) / (scale_max - scale_min + 1e-5)

    return eeg_data


def preprocess_resample(data: np.ndarray, fs: int = 250, refs: int = 125):
    up_factor = refs
    down_factor = fs
    proced = signal.resample_poly(data, up=up_factor, down=down_factor, axis=-1)
    return proced


class PreProcessSequential:
    def __init__(self, config):
        self.fs = config.get('srate', 250)
        self.refs = config.get('re_srate', 125)
        self.bplow = config.get('bp_low', 1)
        self.bphigh = config.get('bp_high', 45)
        self.bslow = config.get('bs_low', 49)
        self.bshigh = config.get('bs_high', 51)
        self.notch = config.get('notch', 50)

    def __call__(self, data: np.ndarray):
        return self._sequential(data)

    def _sequential(self, x):
        x = preprocess_bsfilt(x, low_cut=self.bslow, high_cut=self.bshigh, fs=self.fs)
        x = preprocess_filt(x, low_cut=self.bplow, high_cut=self.bphigh, fs=self.fs)
        x = preprocess_resample(x, fs=self.fs, refs=self.refs)
        x = preprocess_norm(x)
        return x
