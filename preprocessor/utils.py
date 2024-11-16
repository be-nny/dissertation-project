import audio_metadata


def get_song_metadata(path: str) -> str:
    """
    Get the metadata for an audio file. Contains: duration, bitrate, sample_rate

    :param path: path to audio file
    :return: metadata
    """

    metadata = audio_metadata.load(path)
    return f"METADATA '{path}' - duration:{str(metadata['streaminfo']['duration'])}s bitrate:{metadata['streaminfo']['bitrate']}Kbps sample_rate:{metadata['streaminfo']['sample_rate']}Hz"
