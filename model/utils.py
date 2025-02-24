import heapq
import json
import os
import h5py
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial.distance import mahalanobis

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
            self.seg_dur = data['preprocessor_info']['segment_duration']
            self.total_samples = data['preprocessor_info']['total_samples']
            self.created_time = data['start_time']

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


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


class Loader:
    def __init__(self, uuid: str, out: str, logger, batch_size: int = 512):
        self.uuid = uuid
        self.root = os.path.join(out, self.uuid)
        self.logger = logger
        self.input_shape = None
        self.dataloader = None
        self.split_type = None
        self.loaded_files = []
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()

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
        """
        Creates a test or train split. After creation, the data is shuffled randomly.

        :param split_type: type of split (test/train)
        :return: a data split
        """
        split = []
        split_path = os.path.join(self.root, split_type)
        for genre_dir in os.listdir(split_path):
            if not genre_dir.startswith("."):
                genre_path = os.path.join(split_path, genre_dir)
                songs = os.listdir(genre_path)
                for song in songs:
                    if not song.startswith("."):
                        split.append(os.path.join(genre_path, song))

        np.random.shuffle(split)
        return split

    def load(self, split_type: str, normalise: bool = True, genre_filter: list = [], flatten: bool = False) -> DataLoader:
        """
        Loads the preprocessed dataset. The data can be normalised, which normalises the signals by subtracting
        the mean signal and dividing by the standard deviation. The data can also be flattened. This is usually required
        if the data is fed into a dimensionality reduction algorithm like UMAP.

        A genre filter can also be parsed which is a list of string genre tags - this can be empty to load all genres.

        :param split_type: type of split (test/train/all). All will load both the test and train data.
        :param normalise: flag to normalise the signals
        :param genre_filter: the genres to include in the loaded data.
        :param flatten: flag to flatten the dataa
        :return: tensorflow data loader
        """
        self.split_type = split_type
        self.logger.info(f"'normalise' flag set to '{normalise}'")
        self.logger.info(f"'flatten' flag set to '{flatten}'")

        if split_type == "all":
            d1, l1 = self._get_data_split(split_type="test", normalise=normalise, genre_filter=genre_filter)
            d2, l2 = self._get_data_split(split_type="train", normalise=normalise, genre_filter=genre_filter)
            data = np.concatenate((d1, d2), axis=0)
            labels = np.concatenate((l1, l2), axis=0)
        else:
            data, labels = self._get_data_split(split_type=split_type, normalise=normalise, genre_filter=genre_filter)

        data = np.array(data)

        if flatten:
            data = [d.flatten() for d in data]

        int_labels = self.label_encoder.fit_transform(labels)

        data_tensor = torch.tensor(np.array(data), dtype=torch.float32)
        labels_tensor = torch.tensor(int_labels, dtype=torch.int64)

        dataset = TensorDataset(data_tensor, labels_tensor)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self.input_shape = np.array(data[0]).shape
        del data
        return self.dataloader

    def encode_label(self, labels):
        """
        Convert string labels to numeric values

        :param labels: string labels
        :return: numeric values
        """
        return self.label_encoder.transform(labels)

    def decode_label(self, encoded_labels):
        """
        Convert numeric labels back to their original string values

        :param encoded_labels: 1-D array of numeric labels
        :return: string labels
        """
        return self.label_encoder.inverse_transform(encoded_labels)

    def _get_data_split(self, split_type, normalise: bool, genre_filter: list):
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

        if genre_filter != []:
            self.logger.info(f"Loading files with genres:  {', '.join(genre_filter)}")
        else:
            self.logger.info(f"No Genre filters specified. Loading all genres")

        for i in tqdm(range(0, len(split)), unit="file", desc=f"Loading {split_type} data from '{self.uuid}'"):
            with (h5py.File(split[i], "r") as hdf_file):
                b_genre = hdf_file["genre"][()]
                genre = b_genre.decode("utf-8")

                if genre in genre_filter or genre_filter == []:
                    self.loaded_files.append(split[i])
                    signal = np.array(hdf_file["signal"])
                    signal = np.nan_to_num(signal, nan=111111, posinf=222222)

                    signal_data.append(signal)
                    genre_labels.append(genre)

                hdf_file.close()

        signal_data = np.array(signal_data)

        if normalise:
            signal_data = _normalise(signal_data)

        return signal_data, genre_labels

    def get_associated_paths(self):
        """
        :return: The original file path associated with the data points
        """

        return self.loaded_files

    def get_figures_path(self):
        """
        :return: The figures directory inside the UUID's path
        """

        return os.path.join(self.root, 'figures')

class CustomPoint:
    """
    Store additional information for interactive plot
    """
    def __init__(self, point, nearest_neighbours, raw_path, y_pred, y_true):
        self.point = point
        self.x = point[0]
        self.y = point[1]
        self.nearest_neighbours = nearest_neighbours
        self.y_pred = y_pred
        self.y_true = y_true
        self.raw_path = raw_path

def find_nearest_neighbours(latent_space: np.ndarray, point, n_neighbours: int, covar: np.ndarray, raw_paths: list[str] = None, y_true: np.ndarray = None) -> list:
    """
    Uses mahalanobis distance to find the nearest neighbours to a point on the latent space.

    :param y_true: true labels
    :param latent_space: the latent space
    :param raw_paths: the raw .mp3/.wav paths of the original audio
    :param point: a 'CustomPoint' point
    :param n_neighbours: the number of nearest neighbours
    :param covar: covariance matrix
    :return: a list of n nearest neighbours (ascending order)
    """

    nearest_neighbours = []
    for i, latent_point in enumerate(latent_space):
        dist = mahalanobis(point, latent_point, covar)
        if raw_paths is not None:
            path = os.path.basename(raw_paths[i])
            info = (dist, latent_point, path)
        elif y_true is not None:
            info = (dist, latent_point, y_true[i])
        else:
            info = (dist, latent_point)

        nearest_neighbours.append(info)

    sorted_nearest_neighbours = sorted(nearest_neighbours, key=lambda x: x[0])[:n_neighbours+1]

    return sorted_nearest_neighbours

def create_custom_points(latent_space: np.ndarray, raw_paths: list[str], y_pred: np.ndarray, y_true: np.ndarray, covar: np.ndarray, n_neighbours: int = 5) -> list[CustomPoint]:
    """
    Creates a list of 'CustomPoints' to store additional infor about the data points before plotting them.

    :param latent_space: the latent space
    :param raw_paths: the raw .mp3/.wav paths of the original audio
    :param y_pred: predicted labels
    :param y_true: true labels
    :param covar: a list of covariance matrices
    :param n_neighbours: the number of nearest neighbours
    :return: a list of 'CustomPoints'
    """
    custom_points = []

    for i, point in enumerate(latent_space):
        raw_path = None
        curr_y_pred = None
        curr_y_true = None

        if raw_paths is not None:
            raw_path = raw_paths[i]
        if y_pred is not None:
            curr_y_pred = y_pred[i]
        if y_true is not None:
            curr_y_true = y_true[i]

        nearest_neighbours = find_nearest_neighbours(latent_space=latent_space, raw_paths=raw_paths, point=point, n_neighbours=n_neighbours, covar=covar)
        custom_points.append(CustomPoint(point, nearest_neighbours, raw_path, curr_y_pred, curr_y_true))

    return custom_points

def cluster_statistics(y_true: np.ndarray, y_pred: np.ndarray, loader: Loader, logger) -> dict:
    """
    Creates a dictionary containing which clusters have what genre in them. Each genre has a count of the number of samples in that cluster with that genre tag

    :param y_true: true label values
    :param y_pred: predicted label values
    :param loader: dataset loader
    :return: the cluster statistics
    """
    # nmi score
    nmi = normalized_mutual_info_score(y_true, y_pred)
    logger.info(f"NMI score: {nmi}")

    cluster_stats = {}
    # convert the encoded labels back to strings
    y_true = loader.decode_label(y_true)
    for i in range(0, len(y_pred)):
        if y_pred[i] not in cluster_stats:
            cluster_stats.update({y_pred[i]: {}})

        if y_true[i] not in cluster_stats[y_pred[i]]:
            cluster_stats[y_pred[i]].update({y_true[i]: 0})

        cluster_stats[y_pred[i]][y_true[i]] += 1

    return cluster_stats


def correlation(latent_space: np.ndarray, y_true: np.ndarray, covar: np.ndarray, n_neighbours: int = 5):
    """
    Works out the number of 'n' nearest neighbours for every point. Each neighbour is compared to the true label of each point.

    :param latent_space: latent space
    :param y_true: true labels
    :param covar: inverse covariance matrix
    :param n_neighbours: the number of nearest neighbours
    :return: true values, nearest neighbours predicted
    """

    neighbours_true = []
    neighbours_pred = []

    for i, point in enumerate(latent_space):
        p_label = y_true[i]
        neighbours = find_nearest_neighbours(latent_space=latent_space, y_true=y_true, point=point, n_neighbours=n_neighbours, covar=covar)
        for neighbour in neighbours:
            _, p, n_label = neighbour
            neighbours_true.append(p_label)
            neighbours_pred.append(n_label)

    return neighbours_true, neighbours_pred

def connected_graph(latent_space: np.ndarray, covar: np.ndarray, n_neighbours: int = 5) -> dict:
    """
    Creates an undirected weighted graph, given a nearest neighbour value, of every point and what its distance is from
    its nearest points is.

    :param latent_space: the latent space
    :param covar: inverse covariance matrix
    :param n_neighbours: nearest neighbours
    :return: graph
    """

    graph = {}
    for point in latent_space:
        point_str = ','.join(str(p) for p in point)
        neighbour_data = find_nearest_neighbours(latent_space=latent_space, point=point, n_neighbours=n_neighbours, covar=covar)

        conn = []
        for data in neighbour_data:
            dist, neighbour = data
            conn.append((','.join(str(p) for p in neighbour), dist))

        graph.update({point_str: conn})

    return graph

def shortest_path(graph: dict, start:str, end: str):
    """
    Works out the shortest path (using dijkstra's algorithm) between a start and end point given a connect undirected weight graph
    :param graph: undirected weighted graph
    :param start: start node
    :param end: end node

    :return: (distance, shortest_path)
    """

    priority_queue = []

    # (distance, node, path)
    heapq.heappush(priority_queue, (0, start, []))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while priority_queue:
        current_distance, current_node, path = heapq.heappop(priority_queue)
        path = path + [current_node]

        if current_node == end:
            points_path = []
            for node in path:
                points_path.append([np.float64(p) for p in node.split(",")])

            return current_distance, np.array(points_path)

        if current_distance > distances[current_node]:
            continue

        for neighbour, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                heapq.heappush(priority_queue, (distance, neighbour, path))

    return float('inf'), np.array([])
