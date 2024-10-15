import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.io import wavfile
import librosa
import pyloudnorm as pyln
import soundfile as sf


def get_sound_intensity_from_waveform(ir_waveform, specific_acoustic_impedance_air=400):
    # Convert RIR to frequency domain
    freq_rir = fft(ir_waveform)
    # # Air density and speed of sound
    rho = 1.225  # kg/m^3
    c = 343.0    # m/s
    I = (np.abs(freq_rir) ** 2) / (rho * c)
    return np.max(I)


def get_decibels(ir_waveform):
    max_I = get_sound_intensity_from_waveform(ir_waveform)
    max_I = np.clip(max_I, a_min=10**-12, a_max=10**0.8) # TODO: revist this design
    return torch.tensor([10 * np.log10(max_I) + 120])


def load_audio(idx, data_path, split, scene):
    return wavfile.read(os.path.join(data_path, split, scene, f'{idx}-ir_receiver.wav'))[1]


def convert_ir_to_max_db(idx, data_path, split, scene):
    ir_waveform = load_audio(idx, data_path, split, scene)
    max_db = get_decibels(ir_waveform)
    return max_db


def plot_waveform(wavefilename):
    combined = wavefilename.split('/')[-1].split('.')[0]
    action = combined.split('-')[0]
    if action != 'noop':
        speed = combined.split('-')[1][-1]
    else:
        speed = 0
    count = combined.split('-')[-1][1]
    
    # wavfile =  '/home/vidhij/data-for-stretch/data/usb_manno_lavelier/carpet/move_forward-s3-c3.wav'
    audio_data, framerate = librosa.load(wavefilename, sr=None)
    # audio_data = y / (2 ** (sample_width * 8 - 1))

    # Calculate time axis
    time = np.linspace(0, len(audio_data) / framerate, num=len(audio_data))

    # Plot waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_data, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.ylim(-0.2, 0.2)
    plt.title(f"#{count} for {action} at speed {speed}")
    plt.grid()
    plt.show()


def A_weighting(signal, sr):
    A_weighting_curve = librosa.A_weighting(np.fft.rfftfreq(len(signal), 1/sr))
    spectrum = np.fft.rfft(signal)
    weighted_spectrum = spectrum * 10**(A_weighting_curve / 20)
    weighted_signal = np.fft.irfft(weighted_spectrum, n=len(signal))
    return weighted_signal


def C_weighting(signal, sr):
    C_weighting_curve = librosa.C_weighting(np.fft.rfftfreq(len(signal), 1/sr))
    spectrum = np.fft.rfft(signal)
    weighted_spectrum = spectrum * 10**(C_weighting_curve / 20)
    weighted_signal = np.fft.irfft(weighted_spectrum, n=len(signal))
    return weighted_signal


def get_lufs(signal, sr):
    # LUFS calculation using pyloudnorm
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(signal)
    return loudness


def normalize_to_float64(waveform_int16):
    waveform_float64 = waveform_int16.astype(np.float64) / np.iinfo(np.int16).max 
    return waveform_float64


def compute_loudness_metrics_for_robot(robot_audio_files):
    dblog = []
    for wavefilename in robot_audio_files:
        # wavfile =  '/home/vidhij/data-for-stretch/data/usb_manno_lavelier/carpet/move_forward-s3-c3.wav'
        # y, sr = librosa.load(wavefile, sr=44100)
        sr, y = wavfile.read(wavefilename)
        y = normalize_to_float64(y)

        # My own decibel calculation
        max_db = get_decibels(y).item() 
        # print(f"Max dB: {max_db}")

        # LUFS calculation
        lufs = get_lufs(y, sr)
        # print(f"LUFS: {lufs}")

        # RMS calculation
        rms = np.sqrt(np.mean(y**2))
        # print(f"RMS: {rms}")

        # Peak level
        peak = np.max(np.abs(y))
        peak_db = 20 * np.log10(peak)
        # print(f"Peak level: {20 * np.log10(peak)} dB")

        y, sr = librosa.load(wavefilename, sr=44100)
        # A-weighted and C-weighted SPL
        weighted_signal = A_weighting(y, sr)
        spl_a_weighted = 20 * np.log10(np.sqrt(np.mean(weighted_signal**2)) / 20e-6)
        # print(f"A-weighted SPL: {spl_a_weighted} dBA")

        weighted_signal = C_weighting(y, sr)
        spl_c_weighted = 20 * np.log10(np.sqrt(np.mean(weighted_signal**2)) / 20e-6)
        # print(f"C-weighted SPL: {spl_c_weighted} dBC")
        
        dblog.append([wavefilename, max_db, lufs, rms, peak_db, spl_a_weighted, spl_c_weighted])
    df = pd.DataFrame(dblog, columns=['wavefile', 'max_db', 'lufs', 'rms', 'peak_db', 'spl_a_weighted', 'spl_c_weighted'])
    
    # Parse the wavefile to get the action, speed, and count
    df['action_raw'] = df['wavefile'].apply(lambda x: x.split('/')[-1].split('-')[:-2]) 

    df['action'] = df['action_raw'].apply(lambda x: 'noop' if len(x)==0 else x[0])
    df['speed'] = df['wavefile'].apply(lambda x: x.split('/')[-1].split('-')[-2])
    df['count'] = df['wavefile'].apply(lambda x: x.split('/')[-1].split('-')[-1].split('.')[0])
    return df


def get_grouped_loudness_metrics(df):
    grouped_df = df.groupby(['action', 'speed']).agg({key: ['mean', 'std'] for key in ['max_db', 'lufs', 'rms', 'peak_db', 'spl_a_weighted', 'spl_c_weighted']})
    return grouped_df


def load_loudness_metrics(robot_name, dir_path='.'):
    filepath = os.path.join(
        dir_path,
        f'{robot_name}_loudness_metrics.csv'
    )
    return pd.read_csv(
        filepath, 
        index_col=[0, 1], header=[0, 1]
    )

