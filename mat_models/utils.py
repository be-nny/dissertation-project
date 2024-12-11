import os
import h5py
import numpy as np

class Loader:
    def __init__(self, uuid: str, out: str):
        self.uuid = uuid
        self.root = os.path.join(out, self.uuid)

        # creating the test/train arrays
        self.test_split = self._make_splits(split_type="test")
        self.train_split = self._make_splits(split_type="train")

        # getting data from the receipt file
        self.total_samples = len(self.test_split) + len(self.train_split)
        self.genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

        # self.receipt_file = os.path.join(self.root, "receipt.json")
        # with open(self.receipt_file, "r") as f:
        #     self.receipt = json.load(f)
        #     self.total_samples = self.receipt["preprocessor_info"]["total_samples"]
        #     self.genres = self.receipt["genres"]
        #     f.close()

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
        return split

    def get_batch(self, split_type: str, batch_size=10):
        if split_type == "test":
            return self._make_batch(split_arr=self.test_split, batch_size=batch_size)
        elif split_type == "train":
            return self._make_batch(split_arr=self.train_split, batch_size=batch_size)

    def _make_batch(self, split_arr: list, batch_size: int):
        np.random.shuffle(split_arr)
        for i in range(0, self.total_samples, batch_size):
            arr = split_arr[i:i + batch_size]
            data_batch, genre_batch = [], []

            for split in arr:
                with (h5py.File(split, "r") as hdf_file):
                    data_batch.append(np.array(hdf_file["layers"]).flatten())
                    b_genre = hdf_file["genre"][()]
                    genre = b_genre.decode("utf-8")
                    genre_batch.append(genre)

            yield data_batch, genre_batch

    def get_genres(self):
        return self.genres
