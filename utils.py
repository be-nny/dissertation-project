import config
import logging
import os
from model import utils as model_utils

def get_genre_filter(genres_arg: str) -> tuple[int, list]:
    """
    Formats a genre filter string

    :param genres_arg: genre filter from CLI arg
    :return: number of genres, formatted genre list
    """

    if genres_arg != "all":
        genre_filter = genres_arg.replace(" ", "").split(",")
        n_genres = len(genre_filter)
    else:
        genre_filter = []
        n_genres = 10

    return n_genres, genre_filter

def show_info(logger: logging.Logger, config: config.Config) -> None:
    """
    Shows all available datasets to in the output directory in 'config.yml'

    :param logger: logger
    :param config: config file
    """

    datasets = os.listdir(config.OUTPUT_PATH)
    exclude = ["experiments", "gaussian_model", "dec", "analysis"]
    for uuid in datasets:
        if uuid[0] != "." and uuid not in exclude:
            path = os.path.join(config.OUTPUT_PATH, uuid)
            receipt_file = os.path.join(path, "receipt.json")
            with model_utils.ReceiptReader(filename=receipt_file) as receipt_reader:
                out_str = f"{uuid} - {receipt_reader.signal_processor:<15} SAMPLE SIZE: {receipt_reader.total_samples:<5} SEGMENT DURATION:{receipt_reader.seg_dur:<5} CREATED:{receipt_reader.created_time:<10}"

            logger.info(out_str)