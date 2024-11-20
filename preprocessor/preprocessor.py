import os
import shutil

import librosa
from pycparser.ply.cpp import Preprocessor
from tqdm import tqdm

from . import signal_processor
from . import utils

class Preprocessor:
    def __init__(self, dataset_dir, segment_duration=10, output_dir="test_outputs/"):
        """
        Create a preprocessor to preprocess a set of songs in a dataset.

        :param dataset_dir: path to the dataset
        :param segment_duration: the length in seconds for each segment
        :param output_dir: output path of preprocessed files
        """

        self.dataset_dir = dataset_dir
        self.output_dir = output_dir

        # creating output directory for preprocess data
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.mkdir(self.output_dir)

        self._signal_processors = None
        self.segment_duration = segment_duration
        self.reader = utils.DatasetReader(self.dataset_dir)

    def set_signal_processors(self, *signal_processors) -> Preprocessor:
        """
        Set the type of audio spectra that are created for each song

        :param signal_processors: argument list of functions that generate different audio spectra
        :return: current instance
        """

        self._signal_processors = signal_processors
        return self

    def process(self, path: str, genre: str) -> None:
        """
        Preprocesses a song and generates a set of audio spectra specified in `set_layers()`

        :param genre: genre of the song being processed
        :param path: path to audio file
        """
        print(f"\n{utils.get_song_metadata(path=path)}")

        wave, sr = librosa.load(path, sr=None)
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        output_dir = os.path.join(self.output_dir, genre)

        # creating directory for the genre to put the different spectra in
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # creating directory for the song to put the different spectra in
        output_dir = os.path.join(output_dir, name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # creating sub dirs
        for func in self._signal_processors:
            p = os.path.join(output_dir, func.__name__)
            os.mkdir(p)

        # use the signal processor context to create generator that creates the song snippets
        with signal_processor.SignalLoader(wave, sr, segment_duration=self.segment_duration) as loader:
            count = 1
            for segment in loader:
                if len(segment) == 0:
                    break

                # generate the different audio spectra graphs
                for func in self._signal_processors:
                    func(segment, sr, path=f"/{output_dir}/FUNC/{count}.png")
                count +=1

    def preprocess(self):
        """
        Preprocesses the dataset
        """

        for file, genre in tqdm(self.reader, desc="Preprocessing", total=len(self.reader)):
            self.process(file, genre)

    def get_reader(self):
        return self.reader

    def get_segment_duration(self):
        return self.segment_duration

    def get_songs(self):
        return self.reader.files