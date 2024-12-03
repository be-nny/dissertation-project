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
    def __init__(self, dataset_dir, logger, train_split=0.6):
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.files = []
        self.current = 0

        self.train_split = train_split

        self.logger.info("Reading dataset files")
        self._get_files(self.dataset_dir)
        self.logger.info("Completed reading dataset")

        self.files_dict = {}
        for path, genre in self.files:
            if genre.lower() not in self.files_dict:
                self.files_dict.update({genre.lower(): []})

            self.files_dict[genre.lower()].append(path)

        sample_size = self._under_sample()
        self.logger.info(f"Genre sample size: {sample_size}")

        self._test_train_split()
        self.logger.info(f"Train/Test split is {self.train_split*100}:{(1-self.train_split)*100}")

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
        Under samples the dataset so that all genres have the same number of samples as the minimum sample size.
        Chooses k random samples from the files dictionary of genres and songs

        :return: new sample size of each genre
        """
        min_val = min([len(files) for _, files in self.files_dict.items()])

        # choosing k random samples from the files dictionary of genres and songs
        for genre in self.files_dict:
            self.files_dict[genre] = random.sample(self.files_dict[genre], min_val)

        self.total_length = len(self.files_dict) * min_val

        return min_val

    def _test_train_split(self):
        files_dict = {}
        for path, genre in self.files:
            if genre.lower() not in files_dict:
                files_dict.update({genre.lower(): []})

            files_dict[genre.lower()].append(path)

        # creating a test train split
        self.test_train_split = {}
        for genre, files in files_dict.items():
            random.shuffle(files)
            tot = len(files)
            train_num = int(tot * self.train_split)

            self.test_train_split.update({genre: {"train": files[:train_num], "test": files[train_num+1:]}})

    def __enter__(self):
        for genre, splits in self.test_train_split.items():
            train = splits["train"]
            test = splits["test"]

            for path in train:
                yield "train", path, genre

            for path in test:
                yield "test", path, genre

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __len__(self):
        return self.total_length

class JobLogger:
    def __init__(self, uuid_path):
        self.uuid_path = uuid_path
        self.job_log_book = os.path.join(self.uuid_path, "job_log.json")

