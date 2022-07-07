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
def discrete_fourier(audio, save):

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
    if save == True:
        plt.savefig(savefig_path+'/'+'dtf_numpy'+'/'+df['Category'][d]+'-Half.jpg')
    plt.show()

    title = 'Discrete Fourier Transform:'+' '+df['Category'][d]
    plt.plot(magnitude, frequency)
    plt.title(title+'-'+'Full')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    if save == True:
        plt.savefig(savefig_path+'/'+'dtf_numpy'+'/'+df['Category'][d]+'-Full.jpg')
    plt.show()

def spect(audio, n_fft, hop_len, log, save):

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
    plt.title('Spectrogram:'+' '+df['Category'][d])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if save == True:
        plt.savefig(savefig_path+'/'+'spectrograms'+'/'+df['Category'][d]+'.jpg')
    plt.show()

def mel_spect(audio, n_fft, hop_len, save):

    """
    Log Mel Spectrogram

     audio: input audio signal
     n_fft: length of the windowed signal after padding with zeros. The number of rows in the STFT matrix D is (1 + n_fft/2) [1]
     hop_len: number of audio samples between adjacent STFT columns. [1]

     returns: Visualization of mel-spectrogram
    """

    # Signal, sample rate
    sig, sr = audio

    # Transformation
    trans = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_len)
    trans = np.abs(trans)
    trans = librosa.power_to_db(trans, ref=np.max)

    # Visualizations
    plt.figure()
    librosa.display.specshow(trans, cmap='magma', sr=sr, x_axis='time', y_axis='mel', hop_length=hop_len, n_fft=n_fft)
    plt.colorbar(label='Amplitude')
    plt.title('Mel-Spectrogram:'+' '+df['Category'][d])
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if save == True:
        plt.savefig(savefig_path+'/'+'mel_spectrograms'+'/'+df['Category'][d]+'.jpg')
    plt.show()

#----------------------END OF FUNCTIONS-----------------------------------------------------------------------------------#

# Data paths
audio_path = 'urban_sample_data'
meta_path = 'urban8k_samples_meta.csv'
savefig_path = 'fourier_figures'

# Dataframe
df = pd.read_csv(meta_path)

# Discrete Fourier Transform
dtf = input('Discrete Fourier transform?(y/n): ', )

if dtf == 'y':
    save = input('Save figure?(y/n): ', )
    print('Beginning discrette fourier transform...')
    for d in range(len(df['Filename'])):
        print('Start sample: '+str(d+1)+' '+df['Category'][d])
        file = librosa.load(audio_path+'/'+df['Filename'][d])
        if save == 'y':
            discrete_fourier_transform = discrete_fourier(file, save=True)
        else:
            discrete_fourier_transform = discrete_fourier(file, save=False)
    print('DONE')

# Basic Spectrogram
spectrogram = input('Basic spectrogram?(y/n): ', )

if spectrogram == 'y':
    n_fft = int(input('n_fft: ', ))
    hop_len = int(input('Hop length: ', ))
    log = input('Convert amplitude to decibels?(y/n): ', )
    save = input('Save figure?(y/n): ', )
    if log == 'y':
        log = True
    else:
        log = False
    print('Beginning spectrogram...')
    for d in range(len(df['Filename'])):
        print('Start sample: '+str(d+1)+' '+df['Category'][d])
        file = librosa.load(audio_path+'/'+df['Filename'][d])
        if save == 'y':
            spectrogram = spect(file, n_fft=n_fft, hop_len=hop_len, log=log, save=True)
        else:
            spectrogram = spect(file, n_fft=n_fft, hop_len=hop_len, log=log, save=False)
    print('DONE')

# Mel-Spectrogram
mel_spectrogram = input('Mel-spectrogram?(y/n): ', )

if mel_spectrogram == 'y':
    n_fft = int(input('n_fft: ', ))
    hop_len = int(input('Hop length: ', ))
    save = input('Save figure?(y/n): ', )
    print('Beginning Mel-spectrogram...')
    for d in range(len(df['Filename'])):
        print('Start sample: '+str(d+1)+' '+df['Category'][d])
        file = librosa.load(audio_path+'/'+df['Filename'][d])
        if save == 'y':
            mel_spectrogram = mel_spect(file, n_fft=n_fft, hop_len=hop_len, save=True)
        else:
            mel_spectrogram = mel_spect(file, n_fft=n_fft, hop_len=hop_len, save=False)
    print('DONE')

print('All transformations complete.')
