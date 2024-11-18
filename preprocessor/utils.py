import os
import audio_metadata

from audio_metadata import UnsupportedFormat

def get_song_metadata(path: str) -> str:
    """
    Get the metadata for an audio file. Contains: duration, bitrate, sample_rate

    :param path: path to audio file
    :return: metadata
    """

    try:
        metadata = audio_metadata.load(path)
    except UnsupportedFormat:
        return f"METADATA '{path}' - file type not supported"

    return f"METADATA '{path}' - duration:{str(metadata['streaminfo']['duration'])}s bitrate:{metadata['streaminfo']['bitrate']}Kbps sample_rate:{metadata['streaminfo']['sample_rate']}Hz"


class DatasetReader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.files = []
        self.current = 0
        self._get_files(self.dataset_dir)

    def _get_files(self, path):
        directory = os.listdir(path)
        for item in directory:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                self._get_files(item_path)
            else:
                if item_path.endswith(".wav") or item_path.endswith(".mp3"):
                    genre_name = os.path.basename(path).lower()
                    self.files.append((item_path, genre_name))

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
