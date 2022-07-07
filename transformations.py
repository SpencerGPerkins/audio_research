import numpy as np
import pandas as pd

import librosa
import librosa.display
import librosa.feature

import matplotlib.pyplot as plt

"""
This script is for exploring different transformations on audio files.
The purpose is for exploration, experimentation.
At the moment, I do not have any exact planned implementation for these functions.

DATA: UrbanSound8K dataset: 10 samples, one from each class.

[1] in docstrings: taken from librosa documentation
"""

# Functions for different audio transformations
def discrete_fourier(audio):

    """
    Basic Discrete Fourier Transform

    audio: input audio signal

    returns: Visualization of Fast-Fourier transform

    """

    # Signal, sample rate
    sig, sr = audio

    # Create transformation
    trans = np.fft.fft(sig)
    magnitude = np.abs(trans)
    frequency = np.linspace(0, sr, len(magnitude))
    lh_mag = magnitude[:int(len(magnitude)/2)]
    lh_freq = frequency[:int(len(frequency)/2)]

    # Visualizations
    title = 'Discrete Fourier Transform:'+' '+df['Category'][d]
    plt.plot(lh_mag, lh_freq)
    plt.title(title+'-'+'Half')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.savefig(savefig_path+'/'+'dtf_numpy'+'/'+df['Category'][d]+'-Half.jpg')
    plt.show()

    title = 'Discrete Fourier Transform:'+' '+df['Category'][d]
    plt.plot(magnitude, frequency)
    plt.title(title+'-'+'Full')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.savefig(savefig_path+'/'+df['Category'][d]+'-Full.jpg')
    plt.show()

def spect(audio, n_fft, hop_len, log):

    """
     Basic Spectrogram from the Short-time Fourier Transform

     audio: input audio signal
     n_fft: length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2) [1]
     hop_len: number of audio samples between adjacent STFT columns. [1]
     log: boolean, whether to convert from amplitude to decibels

     returns: Visualization of Spectrogram
    """

    # Signal, sample rate
    sig, sr = audio

    # Transformation
    trans = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop_len))

    # Conditional for converting from amplitude to decibels
    if log == True:
        print('Converting amplitude to decibels...')
        trans = librosa.amplitude_to_db(trans)
        y_axis = 'log'
    else:
        y_axis='hz'

    # Visualizations
    plt.figure()
    librosa.display.specshow(trans, cmap='magma', sr=sr, x_axis='time', y_axis=y_axis, hop_length=hop_len)
    plt.colorbar(label='Amplitude')
    plt.title('Spectrogram'+' '+df['Category'][d])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(savefig_path+'/'+'spectrograms'+'/'+df['Category'][d]+'.jpg')
    plt.show()

def mel_spect(audio, n_fft, hop_len):

    """
    Log Mel Spectrogram

     audio: input audio signal
     n_fft: length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2) [1]
     hop_len: number of audio samples between adjacent STFT columns. [1]

     returns: Visualization of mel-spectrogram
    """

    # Signal, sample rate
    sig, sr = audio
    n_fft = 2048
    hop_len = 512

    # Transformation
    trans = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_len)
    trans = np.abs(trans)
    trans = librosa.power_to_db(trans, ref=np.max)

    # Visualizations
    plt.figure()
    librosa.display.specshow(trans, cmap='magma', sr=sr, x_axis='time', y_axis='mel', hop_length=hop_len, n_fft=n_fft)
    plt.colorbar(label='Amplitude')
    plt.title('Mel-Spectrogram'+' '+df['Category'][d])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(savefig_path+'/'+'mel_spectrograms'+'/'+df['Category'][d]+'.jpg')
    plt.show()

#----------------------END OF FUNCTIONS-----------------------------------------------------------------------------------#

# Data paths
audio_path = '/Users/sonic_tertul/Desktop/personal_projs/audio_experiments/urban_sample_data'
meta_path = '/Users/sonic_tertul/Desktop/personal_projs/audio_experiments/urban8k_samples_meta.csv'
savefig_path = '/Users/sonic_tertul/Desktop/personal_projs/audio_experiments/fourier_figures'

# Dataframe
df = pd.read_csv(meta_path)

n_fft = 2048
hop_len = 512

for d in range(len(df['Filename'])):
    file = librosa.load(audio_path+'/'+df['Filename'][d])
    #discrete_fourier_transform = discrete_fourier(file)
    spectrogram = mel_spect(file, n_fft=n_fft, hop_len=hop_len)

print('DONE')
