import json
import os
import h5py
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

class ReceiptReader:
    def __init__(self, filename):
        self.filename = filename
        self.genres = []
        self.signal_processor = []

    def __enter__(self):
        with open(self.filename, 'r') as f:
            data = json.load(f)
            self.genres = data['genres']
            self.signal_processor = data['preprocessor_info']['signal_processor']

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

class Loader:
    def __init__(self, uuid: str, out: str, logger, batch_size: int = 512):
        self.uuid = uuid
        self.root = os.path.join(out, self.uuid)
        self.logger = logger
        self.input_shape = None
        self.batch_size = batch_size

        # creating the test/train arrays
        self.test_split = self._make_splits(split_type="test")
        self.train_split = self._make_splits(split_type="train")

        # getting data from the receipt file
        self.total_samples = len(self.test_split) + len(self.train_split)

        with ReceiptReader(filename=os.path.join(self.root, 'receipt.json')) as receipt:
            self.genres = receipt.genres
            self.signal_processor = receipt.signal_processor

        self.logger.info(f"'{self.uuid}' applied with {self.signal_processor}")

    def _make_splits(self, split_type: str) -> list:
        split = []
        train_path = os.path.join(self.root, split_type)
        for genre_dir in os.listdir(train_path):
            if not genre_dir.startswith("."):
                genre_path = os.path.join(train_path, genre_dir)
                songs = os.listdir(genre_path)
                for song in songs:
                    if not song.startswith("."):
                        split.append(os.path.join(genre_path, song))

        np.random.shuffle(split)
        return split

    def load(self, split_type: str, normalise: bool = True):
        if split_type == "all":
            d1, l1 = self._get_data_split(split_type="test", normalise=normalise)
            d2, l2 = self._get_data_split(split_type="train", normalise=normalise)
            data = np.concatenate((d1, d2), axis=0)
            labels = np.concatenate((l1, l2), axis=0)
        else:
            data, labels = self._get_data_split(split_type=split_type, normalise=normalise)

        data = np.array(data)

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(labels)

        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(int_labels, dtype=torch.int64)

        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.input_shape = np.array(data[0]).shape

        return dataloader

    def _get_data_split(self, split_type, normalise: bool):
        """
        This returns a shuffled dataset containing either test or train data from a dataset. This returns an array
        (num_samples, num_features) that are normalised using decimal scaling, and the genre tags (num_samples,) as
        strings.
        :param split_type: return an array that contains either test or train data
        :return: dataset, genres
        """

        if split_type == "test":
            split = self.test_split
        elif split_type == "train":
            split = self.train_split
        else:
            raise ValueError("split_type must be either 'train' or 'test'")

        signal_data = []
        genre_labels = []

        for i in tqdm(range(0, len(split)), unit="file", desc=f"Loading {split_type} data from '{self.uuid}'"):
            with (h5py.File(split[i], "r") as hdf_file):
                signal = np.array(hdf_file["signal"])
                b_genre = hdf_file["genre"][()]
                genre = b_genre.decode("utf-8")

                signal_data.append(signal)
                genre_labels.append(genre)

                hdf_file.close()

        signal_data = np.array(signal_data)

        if normalise:
            signal_data = self._normalise(signal_data)

        return signal_data, genre_labels

    @staticmethod
    def _normalise(signal_data: np.array) -> np.array:
        """
        Normalises the signals by subtracting the mean signal and dividing by the standard deviation. The mean signal
        is a 2D array, and the standard deviation is a single scalar.

        :param signal_data: signal data to be normalised (shape: [s,n,m])
        :return: normalised signal of size (shape: [s,n,m])
        """

        std = np.std(signal_data)
        mean = np.mean(signal_data, axis=0)

        normalised_signal = (signal_data - mean) / std

        return normalised_signal

    def get_figures_path(self):
        """
        :return: The figures directory inside the UUID's path
        """

        return os.path.join(self.root, 'figures')
