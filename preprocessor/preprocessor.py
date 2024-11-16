import os
import shutil
import librosa
from pycparser.ply.cpp import Preprocessor

from . import signal_processor

class Preprocessor:
    def __init__(self, segment_duration=10):
        """
        Create a song preprocessor

        :param segment_duration: duration of the song segments for every song
        """
        self.OUTPUT_DIR = "test_outputs/"
        self._filters = None
        self.segment_duration = segment_duration

    def set_signal_filters(self, *filters) -> Preprocessor:
        """
        Set the type of audio spectra that are created for each song

        :param filters: argument list of functions that generate different audio spectra
        :return: current instance
        """

        self._filters = filters
        return self

    def process(self, path: str) -> None:
        """
        Preprocesses a song and generates a set of audio spectra specified in `set_layers()`

        :param path: path to audio file
        """

        wave, sr = librosa.load(path, sr=None)
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)

        output_dir = os.path.join(self.OUTPUT_DIR, name)

        # creating directory to put the different spectra in
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.mkdir(output_dir)

        # creating sub dirs
        for func in self._filters:
            p = os.path.join(output_dir, func.__name__)
            os.mkdir(p)

        # use the signal processor context to create generator that creates the song snippets
        with signal_processor.SignalLoader(wave, sr, name=path, segment_duration=self.segment_duration) as loader:
            count = 1
            for segment in loader:
                if len(segment) == 0:
                    break

                # generate the different audio spectra graphs
                for func in self._filters:
                    func(segment, sr, path=f"{output_dir}/FUNC/{count}.png")
                count +=1
