"""
This file is responsible for generating a selection of audio spectra for a given audio source. Each spectrogram is
defined in its own method and also generates an example spectrogram when 'debug' is set to True. The example figures
are saved to the 'figures' directory that is created when a new preprocessed dataset is created.
"""

import librosa
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from scipy import signal

matplotlib.use('TkAgg')
plt.figure()
plt.close()

FIGURE_SIZE = (10, 10)
N_FFT = 512
NPERSEG = 256
HOP_LENGTH = 512
N_MELS = 128

class SignalLoader:
    def __init__(self, wave: np.ndarray, sr: float, segment_duration=10):
        """
        Create a Signal Loader Context to create a set of segments from a song.

        :param wave: raw audio data
        :param sr: sample rate
        :param segment_duration: length of each song snippet
        """

        self.wave = wave
        self.sr = sr
        self.segment_duration = segment_duration
        self._number_segments = 0

    def __enter__(self):
        """
        Generator for creating a set of segments from a song.
        :return: song segment
        """

        segment_length = self.sr * self.segment_duration
        for segment in range(0, len(self.wave), segment_length):
            wav = self.wave[segment: segment + segment_length]
            # padding the array with zeros if the wav form wav doesn't match the segment length
            if len(wav) != segment_length:
                wav = np.pad(wav, (0, segment_length - len(wav)))
            self._number_segments += 1
            yield wav

    def __exit__(self, type, value, traceback):
        pass

def STFT(wave: np.ndarray, sr: float, path=None, debug=False):
    """
    Create a Short Time Fourier Transform for a waveform.
    This a method for analyzing how the frequency content of a signal changes over time.
    The initial signal is broken down into small discrete samples of time before having a Fourier Transform applied to it.


    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path, default to None
    :param debug: debug mode, default to False. If true, an example figure will be saved
    :return: an array of magnitudes of each complex value in the Short-Time Fourier Transform (STFT) result.
    """

    # if the length of the wav is smaller than the window function, stop
    if len(wave) < NPERSEG:
        return

    f, t, transform = signal.stft(wave, fs=sr, nperseg=NPERSEG, nfft=N_FFT)

    if not debug:
        # transform contains complex values, the complex components contain phase
        # information - taking the magnitude results in only amplitude.
        return list(np.abs(transform))
    else:
        plt.figure(figsize=FIGURE_SIZE)
        plt.pcolormesh(t, f, np.abs(transform), shading='gouraud')
        plt.title("Example STFT Graph")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return None

def MEL_SPEC(wave: np.ndarray, sr: float, path=None, debug=False):
    """
    Create a Mel Spectrogram for a waveform.
    This is a visual representation of the frequency spectrum of an audio signal over time, where the frequencies are converted to the mel scale
    It displays the intensity of different frequency components in an audio signal.

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path, default to None
    :param debug: debug mode, default to False. If true, an example figure will be saved
    :return: mel spectrogram on a logarithmic scale
    """

    # if the length of the wav is smaller than the window function, stop
    if len(wave) < N_FFT:
        return

    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)

    # this converts the power values to a logarithmic scale since humans perceive loudness this way
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if not debug:
        return mel_spectrogram_db
    else:
        # Plot the Mel spectrogram
        plt.figure(figsize=FIGURE_SIZE)
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.title(f"Example MEL_SPEC Graph")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return None


def CQT(wav: np.ndarray, sr: float, path=None, debug=False):
    """
    Generate the constant-Q transform of an audio signal.
    This converts a data series from the time domain to the frequency domain.
    The CQT's output bins are spaced logarithmically in frequency, unlike the short-time Fourier transform, which uses linear spacing.

    :param wav: raw audio data
    :param sr: sample rate
    :param path: output path, default to None
    :param debug: debug mode, default to False
    :return: cqt as an image array
    """

    cqt_transform = np.abs(librosa.cqt(wav, sr=sr, hop_length=HOP_LENGTH))

    # converted to a logarithmic scale
    cqt_transform_db = librosa.amplitude_to_db(cqt_transform, ref=np.max)

    if not debug:
        return cqt_transform_db
    else:
        plt.figure(figsize=FIGURE_SIZE)
        librosa.display.specshow(cqt_transform_db, sr=sr, x_axis='time', y_axis='cqt_note')
        plt.title(f"Example CQT Graph")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return None

def SPEC_CENTROID(wav: np.ndarray, sr: float, path=None, debug=False):
    """
    Generate a spectral centroid of the audio file.
    A spectral centroid is the location of the centre of mass of the spectrum.

    :param wav: raw audio data
    :param sr: sample rate
    :param path: output path, default to None
    :param debug: debug mode, default to False
    :return: spectral centroid as an image array
    """

    S, _ = librosa.magphase(librosa.stft(y=wav))
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr, hop_length=HOP_LENGTH)

    if not debug:
        return spectral_centroid
    else:
        plt.figure(figsize=FIGURE_SIZE)
        plt.title(f"Example SPECTRAL CENTROID Graph")
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
        times = librosa.frames_to_time(np.arange(spectral_centroid.shape[1]), sr=sr)

        plt.plot(times, spectral_centroid[0], color='white', label='Spectral Centroid')
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return None

def MFCC(wav: np.ndarray, sr: float, path=None, debug=False):
    """
    Generate a MFCC of the audio file.

    :param wav: raw audio data
    :param sr: sample rate
    :param path: output path, default to None
    :param debug: debug mode, default to False
    :return: spectral centroid as an image array
    """

    mfcc = librosa.feature.mfcc(y=wav, sr=sr, hop_length=HOP_LENGTH, n_mels=N_MELS, n_fft=N_FFT)
    mfcc_db = librosa.power_to_db(mfcc, ref=np.max)

    if not debug:
        return mfcc_db
    else:
        plt.figure(figsize=FIGURE_SIZE)
        librosa.display.specshow(mfcc_db, sr=sr, hop_length=HOP_LENGTH, x_axis='time')
        plt.title(f"Example MFCC Graph")
        plt.colorbar(format="%+2.0f dB")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()
        return None

def get_type(name: str):
    """
    Returns the function signature from a given name of a signal processor

    :param name: name of signal processor
    :return: function signature
    """

    if name == CQT.__name__:
        return CQT
    if name == STFT.__name__:
        return STFT
    if name == MEL_SPEC.__name__:
        return MEL_SPEC
    if name == SPEC_CENTROID.__name__:
        return SPEC_CENTROID
    if name == MFCC.__name__:
        return MFCC

    raise ValueError(f"Unknown signal processor: {name}")

def get_all_types():
    """
    :return: All signal processors as a list of their names
    """
    return [CQT.__name__, STFT.__name__, MEL_SPEC.__name__, SPEC_CENTROID.__name__, MFCC.__name__]