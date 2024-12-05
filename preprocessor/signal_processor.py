import io
import librosa
import matplotlib
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from scipy import signal

matplotlib.use('TkAgg')
plt.figure()
plt.close()

FIGURE_SIZE = (10, 10)

class SignalLoader:
    def __init__(self, wave: np.ndarray, sr: float, segment_duration=10):
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
    :return: stft as an image array
    """

    # if the length of the wav is smaller than the window function, stop
    nperseg = 256
    if len(wave) < nperseg:
        return

    f, t, Zxx = signal.stft(wave, fs=sr, nperseg=nperseg)

    # plot the stft
    plt.figure(figsize=FIGURE_SIZE)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.ylim(0, 5000)

    if not debug:
        plt.axis("off")
        # creating buffer and writing to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # creating Image from buffer
        img = Image.open(buf)
        plt.close()
        return img

    else:
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
    :param debug: debug mode, default to False
    :return: mel spectrogram as an image array
    """

    # if the length of the wav is smaller than the window function, stop
    n_fft = 2048
    if len(wave) < n_fft:
        return

    mel_spectrogram = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=512, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel spectrogram
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    if not debug:
        plt.axis("off")

        # creating buffer and writing to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # creating Image from buffer
        img = Image.open(buf)
        plt.close()
        return img
    else:
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

    C = np.abs(librosa.cqt(wav, sr=sr))
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max), sr=sr, x_axis='time', y_axis='cqt_note')
    if not debug:
        plt.axis("off")

        # creating buffer and writing to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # creating Image from buffer
        img = Image.open(buf)
        plt.close()
        return img
    else:
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
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    plt.figure(figsize=FIGURE_SIZE)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')

    times = librosa.frames_to_time(np.arange(spectral_centroid.shape[1]), sr=sr)
    plt.plot(times, spectral_centroid[0], color='white', label='Spectral Centroid')

    if not debug:
        plt.axis("off")

        # creating buffer and writing to the buffer
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0)
        buf.seek(0)

        # creating Image from buffer
        img = Image.open(buf)
        plt.close()
        return img
    else:
        plt.title(f"Example SPECTRAL CENTROID Graph")
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

    raise ValueError(f"Unknown signal processor: {name}")

def get_all_types():
    """
    :return: All signal processors as a list of their names
    """
    return [CQT.__name__, STFT.__name__, MEL_SPEC.__name__, SPEC_CENTROID.__name__]