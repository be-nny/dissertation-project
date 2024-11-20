import time
import librosa
import matplotlib
import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt
from scipy import signal

matplotlib.use('TkAgg')
plt.figure()
plt.close()

FIGURE_SIZE = (10, 10)

class SignalLoader:
    def __init__(self, wave: np.ndarray, sr: float, name: str, segment_duration=10):
        """
        Create a Signal Loader Context to create a set of segments from a song.

        :param wave: raw audio data
        :param sr: sample rate
        :param name: name of the audio file
        :param segment_duration: length of each song snippet
        """

        self.wave = wave
        self.sr = sr
        self.segment_duration = segment_duration
        self.song_name = name
        self._number_segments = 0

    def __enter__(self):
        """
        Generator for creating a set of segments from a song.
        :return: song segment
        """
        segment_length = self.sr * self.segment_duration
        for segment in tqdm(range(0, len(self.wave), segment_length), colour="green", desc="Segmenting"):
            wav = self.wave[segment: segment + segment_length]
            # padding the array with zeros if the wav form wav doesn't match the segment length
            if len(wav) != segment_length:
                wav = np.pad(wav, (0, segment_length - len(wav)))
            self._number_segments += 1
            yield wav

    def __exit__(self, type, value, traceback):
        pass

def STFT(wave: np.ndarray, sr: float, path: str) -> None:
    """
    Create a Short Time Fourier Transform for a waveform

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path
    """

    # if the length of the wav is smaller than the window function, stop
    nperseg = 256
    if len(wave) < nperseg:
        return

    f, t, Zxx = signal.stft(wave, fs=sr, nperseg=nperseg)

    path = path.replace("FUNC", "STFT")

    # plot the stft
    plt.figure(figsize=FIGURE_SIZE)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.ylim(0, 5000)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

def MEL_SPEC(wave: np.ndarray, sr: float, path: str) -> None:
    """
    Create a Mel Spectrogram for a waveform

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path
    """

    # if the length of the wav is smaller than the window function, stop
    n_fft = 2048
    if len(wave) < n_fft:
        return

    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=512, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    path = path.replace("FUNC", "MEL_SPEC")

    # Plot the Mel spectrogram
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def CQT(wav: np.ndarray, sr: float, path: str):
    """
    Generate the constant-Q transform of an audio signal

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path
    """
    path = path.replace("FUNC", "CQT")

    C = np.abs(librosa.cqt(wav, sr=sr))
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
