import os
import random
import audio_metadata
import librosa
from audioread import NoBackendError
from tqdm import tqdm

from audio_metadata import UnsupportedFormat

def create_graph_example_figures(*signal_processors, song_paths, figures_path, num_songs=3) -> None:
    """
    Creates a set of example figures for a given song. This uses the entire duration of the song

    :param signal_processors: the signal processor functions to apply to the wave form
    :param song_paths: list of all songs
    :param figures_path: path to save figures
    :param num_songs: number of random figures to generate
    """

    for i in tqdm(range(0, num_songs), desc="Creating Example Figures"):
        path, genre = random.choice(song_paths)

        wave, sr = librosa.load(path, sr=None)
        for func in signal_processors:
            func(wave, sr, path=f"{figures_path}/{genre}_example_figure_{func.__name__}_{i}.pdf", debug=True)

def get_song_metadata(path: str) -> str:
    """
    Get the metadata for an audio file. Contains: duration, bitrate, sample_rate

    :param path: path to audio file
    :return: metadata
    """

    try:
        metadata = audio_metadata.load(path)
    except UnsupportedFormat:
        return f"METADATA '{path}' - could not load file"

    return f"METADATA '{path}' - duration:{str(metadata['streaminfo']['duration'])}s bitrate:{metadata['streaminfo']['bitrate']}Kbps sample_rate:{metadata['streaminfo']['sample_rate']}Hz"


class DatasetReader:
    def __init__(self, dataset_dir, logger):
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.files = []
        self.current = 0

        self.logger.info("Reading dataset files")
        self._get_files(self.dataset_dir)
        self.logger.info("Completed reading dataset!")

        # shuffle the dataset
        random.shuffle(self.files)

        self.logger.info("Under sampling dataset files")
        sample_size = self._under_sample()
        self.logger.info(f"Completed under sampling dataset with sample size: {sample_size}!")

    def _get_files(self, path: str) -> None:
        """
        Recursive function to read all the files in the dataset in each genre directory

        :param path: path to dataset or path in dataset during recursion
        """

        directory = os.listdir(path)
        for item in directory:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                self._get_files(item_path)
            else:
                if item_path.endswith(".wav") or item_path.endswith(".mp3"):
                    genre_name = os.path.basename(path).lower()
                    try:
                        # try and load the file as this is what the preprocessing loads with
                        # if it fails, don't add the song to the list of files
                        librosa.load(item_path, sr=None)
                        self.files.append((item_path, genre_name))
                    except NoBackendError:
                        self.logger.warning(f"Could not read file '{item_path}', skipping")

    def _under_sample(self):
        """
        Under samples the dataset so that all genres have the same number of samples as the minimum sample size

        :return: new sample size of each genre
        """

        genre_dict = {}

        for path, genre in self.files:
            # adding a count to each genre
            if genre not in genre_dict:
                genre_dict.update({genre: 1})
            else:
                genre_dict[genre] += 1

        min_genre_val = genre_dict[min(genre_dict, key=genre_dict.get)]

        tmp = self.files
        for path, genre in tmp:
            num_genre = genre_dict[genre]
            if num_genre > min_genre_val:
                self.files.remove((path, genre))

        return min_genre_val

    def __next__(self):
        if self.current < len(self.files):
            current_path = self.files[self.current]
            self.current += 1
            return current_path[0], current_path[1]
        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

class JobLogger:
    def __init__(self, uuid_path):
        self.uuid_path = uuid_path
        self.job_log_book = os.path.join(self.uuid_path, "job_log.json")

