import time
import librosa
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from scipy import signal

matplotlib.use('TkAgg')

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
        self.start_time = 0
        self.stop_time = 0

    def __enter__(self):
        """
        Generator for creating a set of segments from a song.
        :return: song segment
        """
        self.start_time = time.time()
        print(f"+ creating segments for {self.song_name}")
        segment_length = self.sr * self.segment_duration
        for s in range(0, len(self.wave), segment_length):
            t = self.wave[s: s + segment_length]
            self._number_segments += 1
            yield t

    def __exit__(self, type, value, traceback):
        self.stop_time = time.time()
        print(f"* created {self._number_segments} segments in {self.stop_time-self.start_time:.2f} seconds")
        pass

def STFT(wave: np.ndarray, sr: float, path: str) -> None:
    """
    Create a Short Time Fourier Transform for a waveform

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path
    """

    f, t, Zxx = signal.stft(wave, fs=sr)

    path = path.replace("FUNC", "STFT")

    # plot the stft
    plt.figure(figsize=FIGURE_SIZE)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.ylim(0, 5000)
    plt.axis("off")
    plt.savefig(path)
    plt.close()

def MEL_SPEC(wave: np.ndarray, sr: float, path: str) -> None:
    """
    Create a Mel Spectrogram for a waveform

    :param wave: raw audio data
    :param sr: sample rate
    :param path: output path
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    path = path.replace("FUNC", "MEL_SPEC")

    # Plot the Mel spectrogram
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.axis("off")
    plt.savefig(path)
    plt.close()


def CQT():
    pass
