import json
import os
import h5py
import matplotlib
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
    def __init__(self, uuid: str, out: str, logger):
        self.uuid = uuid
        self.root = os.path.join(out, self.uuid)
        self.logger = logger
        self.input_shape = None

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

    def get_input_shape(self):
        return self.input_shape

    def get_data(self):
        data, genre_labels = self.get_data_split(split_type='train')
        tmp_d, tmp_g = self.get_data_split(split_type='test')

        data.extend(tmp_d)
        genre_labels.extend(tmp_g)

        return data, genre_labels

    def get_dataloader(self, split_type: str, batch_size: int = 512):
        data, labels = self.get_data_split(split_type=split_type)

        data = np.array(data)

        label_encoder = LabelEncoder()
        int_labels = label_encoder.fit_transform(labels)

        data_tensor = torch.tensor(data, dtype=torch.float32)
        labels_tensor = torch.tensor(int_labels, dtype=torch.int64)

        dataset = TensorDataset(data_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        self.input_shape = np.array(data[0]).shape

        return dataloader

    def get_data_split(self, split_type):
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
                layers = np.array(hdf_file["signal"])
                b_genre = hdf_file["genre"][()]
                genre = b_genre.decode("utf-8")

                # removing any nan values
                layers = np.nan_to_num(layers, nan=0.0, posinf=1e9, neginf=-1e9)

                signal_data.append(layers)
                genre_labels.append(genre)

                hdf_file.close()

        signal_data = np.array(signal_data)
        return signal_data, genre_labels

    def get_genres(self):
        return self.genres

    def get_directory(self):
        return self.root

    def get_figures_path(self):
        return os.path.join(self.root, 'figures')
