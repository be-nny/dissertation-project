import datetime
import json
import os
import uuid
import h5py
import librosa
import math
import numpy as np
import logging

from audioread import NoBackendError
from pycparser.ply.cpp import Preprocessor
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer, MinMaxScaler
from tqdm import tqdm

from . import signal_processor
from . import utils

class Preprocessor:
    def __init__(self, dataset_dir: str, output_dir: str, target_length: int, logger: logging.Logger, train_split=0.6, segment_duration=10):
        """
        Create a preprocessor to preprocess a set of songs in a dataset.

        :param dataset_dir: path to the dataset
        :param segment_duration: the length in seconds for each segment
        :param output_dir: output path of preprocessed files
        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.logger = logger

        self.start_datetime = None
        self.end_time = None

        # creating output directory for preprocess data
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # create unique dir name
        self.uuid = str(uuid.uuid4().hex)[:6]
        self.logger.info(f"'{self.uuid}' created")
        self.uuid_path = os.path.join(self.output_dir, self.uuid)
        os.mkdir(self.uuid_path)

        # adding 'train' to output path
        self.train_split = os.path.join(self.uuid_path, "train")
        os.mkdir(self.train_split)

        # adding 'test' to output path
        self.test_split = os.path.join(self.uuid_path, "test")
        os.mkdir(self.test_split)

        # add 'figures' to output path
        self.figures_path = os.path.join(self.uuid_path, "figures")
        os.mkdir(self.figures_path)

        self.signal_processor = None
        self.segment_duration = segment_duration
        self.target_length = target_length

        self.reader = utils.DatasetReader(self.dataset_dir, self.logger, train_split=train_split)
        self.total = 0

    def set_signal_processor(self, signal_processor) -> Preprocessor:
        """
        Set the type of audio spectra that are created for each song

        :param signal_processors: argument list of functions that generate different audio spectra
        :return: current instance
        """

        self.signal_processor = signal_processor
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
        :param rms: rms level in dB
        :return: normalised wave
        """

        rms_original = np.sqrt(np.mean(wave ** 2))
        scale_factor = rms / rms_original
        wave_normalised = wave * scale_factor
        return wave_normalised

    def _process(self, path: str, genre: str, split_type: str) -> None:
        """
        Preprocesses a song and generates a set of audio spectra specified in `set_layers()` (see -h, --help for more info)

        :param genre: genre of the song being processed
        :param path: path to audio file
        """

        # load wave source
        wave, sr = librosa.load(path, sr=None)

        # normalise the length of the audio file
        wave = self._normalise_length(wave, sr)
        wave = self._rms_normalise_audio(wave)

        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        if split_type == 'train':
            output_dir = os.path.join(self.train_split, genre)
        else:
            output_dir = os.path.join(self.test_split, genre)

        # creating directory for the genre to put the different spectra in
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # use the signal processor context to create generator that creates the song snippets
        with signal_processor.SignalLoader(wave, sr, segment_duration=self.segment_duration) as loader:
            count = 1
            for segment in loader:
                if len(segment) == 0:
                    break

                # generate the desired audio spectrogram
                raw_signal = self.signal_processor(segment, sr)

                scalar = MinMaxScaler(feature_range=(-3, 3))
                raw_signal = scalar.fit_transform(raw_signal)

                # save the layers to HDF5 file
                file_name = os.path.join(output_dir, f"{name}_{count}.h5")
                self._create_hdf(path=file_name, signal=raw_signal, genre=genre)

                # properly discard the layers arr
                del raw_signal
                self.total += 1
                count += 1

    def preprocess(self):
        """
        Preprocesses the dataset with the specified configuration file
        """

        self.logger.info("Preprocessing dataset...")
        self.start_datetime = datetime.datetime.now()

        with self.reader as r:
            for split_type, file, genre in tqdm(r.generate(), desc="Generating Spectra", total=len(self.reader), unit="file"):
                try:
                    self._process(file, genre, split_type)
                except NoBackendError:
                    self.logger.warning(
                        f"Could not read file '{file}' with META DATA `{utils.get_song_metadata(path=file)}' skipping")
                    continue

        # write receipt file
        self.write_receipt()

    def write_receipt(self):
        """
        Writes a receipt file after preprocessing is complete

        :return:
        """

        self.logger.info("Writing receipt...")

        json_data = {
            "uuid": self.uuid,
            "genres": self.reader.get_total_genres(),
            "start_time": str(self.start_datetime),
            "end_time": str(datetime.datetime.now()),
            "preprocessor_info": {
                "segment_duration": self.segment_duration,
                "target_length": self.target_length,
                "total_samples": self.total,
                "signal_processor": str(self.signal_processor.__name__),
                "n_fft": signal_processor.N_FFT,
                "hop_length": signal_processor.HOP_LENGTH,
                "n_mels": signal_processor.N_MELS,
            }
        }

        receipt_path = os.path.join(self.uuid_path, "receipt.json")

        with open(receipt_path, 'w') as f:
            json.dump(json_data, f)

    def get_dataset_reader(self):
        """
        :return: Dataset reader
        """

        return self.reader

    def get_songs(self) -> list:
        """
        :return: The list of songs
        """

        return self.reader.files

    def get_signal_processor(self) -> list:
        """
        :return: A list of signal processors
        """

        return self.signal_processor

    def get_figures_path(self) -> str:
        """
        Get the figures path

        :return: figures path
        """

        return self.figures_path
