import os
import uuid
import h5py
import librosa
import math
import numpy as np

from audioread import NoBackendError
from pycparser.ply.cpp import Preprocessor
from tqdm import tqdm

from . import signal_processor
from . import utils


class Preprocessor:
    def __init__(self, dataset_dir, output_dir, target_length, segment_duration=10):
        """
        Create a preprocessor to preprocess a set of songs in a dataset.

        :param dataset_dir: path to the dataset
        :param segment_duration: the length in seconds for each segment
        :param output_dir: output path of preprocessed files
        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        # creating output directory for preprocess data
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # create unique dir name
        new_dir_name = str(uuid.uuid4().hex)
        print(f"'{new_dir_name}' created")
        new_path = os.path.join(self.output_dir, new_dir_name)
        self.output_dir = new_path
        os.mkdir(self.output_dir)

        self._signal_processors = None
        self.segment_duration = segment_duration
        self.target_length = target_length
        self.reader = utils.DatasetReader(self.dataset_dir)

    def set_signal_processors(self, *signal_processors) -> Preprocessor:
        """
        Set the type of audio spectra that are created for each song

        :param signal_processors: argument list of functions that generate different audio spectra
        :return: current instance
        """

        self._signal_processors = signal_processors
        return self

    def _create_hdf(self, path: str, **kwargs):
        """
        Construct a HDF5 file for a preprocessed song and save it to 'path'.

        :param path: path to save the HDF5 file
        :param kwards: key value pairs for the hdf file
        :return:
        """

        with h5py.File(path, 'w') as hdf5_file:
            for key, data in kwargs.items():
                hdf5_file.create_dataset(key, data=data)

    def _normalise_length(self, wave, sr):
        """
        Normalise the length of a waveform so that it is the same length as the target length

        :param wave: waveform to normalise
        :param sr: the sample rate
        :return: a waveform that is the same length as the target length and padded if necessary
        """

        duration = librosa.get_duration(y=wave, sr=sr)
        diff = self.target_length - duration
        diff_samples = math.fabs(diff * sr)

        # if the number of samples is larger than the target length, truncate it down
        # if it's smaller the SignalProcessor will pad the wav
        if diff < 0:
            new_samples = int(len(wave) - diff_samples)
            wave = wave[:new_samples]

        return wave

    def _rms_normalise_audio(self, wave, rms=0.1):
        """
        Root Mean Squared (RMS) audio normalisation. Balances the perceived loudness to create a cohesive
        sound across all sources.

        :param wave: input wave
        :return: normalised wave
        """

        rms_original = np.sqrt(np.mean(wave ** 2))
        scale_factor = rms / rms_original
        wave_normalised = wave * scale_factor
        return wave_normalised

    def process(self, path: str, genre: str) -> None:
        """
        Preprocesses a song and generates a set of audio spectra specified in `set_layers()` (see -h, --help for more info)

        :param genre: genre of the song being processed
        :param path: path to audio file
        """

        print(f"\n{utils.get_song_metadata(path=path)}")

        # load wave source
        wave, sr = librosa.load(path, sr=None)

        # normalise the length of the audio file
        wave = self._normalise_length(wave, sr)
        wave = self._rms_normalise_audio(wave)

        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        output_dir = os.path.join(self.output_dir, genre)

        # creating directory for the genre to put the different spectra in
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # use the signal processor context to create generator that creates the song snippets
        with signal_processor.SignalLoader(wave, sr, segment_duration=self.segment_duration) as loader:
            count = 1
            for segment in loader:
                if len(segment) == 0:
                    break

                # generate the different audio spectra graphs
                layers = []
                for func in self._signal_processors:
                    img = func(segment, sr)
                    arr = np.asarray(img)
                    layers.append(arr)

                # save the layers to HDF5 file
                file_name = os.path.join(output_dir, f"{name}_{count}.h5")
                self._create_hdf(path=file_name, layers=layers)

                count += 1

    def preprocess(self):
        """
        Preprocesses the dataset
        """

        print("Preprocessing dataset...")
        for file, genre in tqdm(self.reader, desc="Generating Spectra", total=len(self.reader)):
            try:
                self.process(file, genre)
            except NoBackendError:
                print(f"WARNING - could not read file '{file}', skipping")
                continue

    def get_reader(self):
        """
        :return: The dataset reader
        """
        return self.reader

    def get_segment_duration(self):
        """
        :return: The length in seconds for each segment
        """

        return self.segment_duration

    def get_songs(self):
        """
        :return: The list of songs
        """
        return self.reader.files

    def get_signal_processors(self):
        """
        :return: A list of signal processors
        """
        return self._signal_processors
