import os
import argparse

from preprocessor import preprocessor as p
from preprocessor import signal_processor as sp
from preprocessor import utils as pu
from dotenv import load_dotenv

# arguments parser
parser = argparse.ArgumentParser(prog='PROG',description="Music Analysis Tool")
parser.add_argument("-p", "--process", action="store_true", help="Preprocess data")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-f", "--figures", action="store", default=1, type=int, help="Create a set of n example figures")

if __name__ == "__main__":
    args = parser.parse_args(["--process", "--config=config.env"])

    # load config file
    if args.config:
        load_dotenv(dotenv_path=args.config)
        dataset_path = os.getenv("DATASET_PATH")
        output_path = os.getenv("PREPROCESSED_PATH")
        figures_path = os.getenv("FIGURES_PATH")

    if dataset_path is None or output_path is None or figures_path is None:
        raise ValueError(f".env file has missing arguments! got: DATASET_PATH='{dataset_path}', OUTPUT_PATH='{output_path}', FIGURES_PATH='{figures_path}'")

    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=dataset_path, target_length=30, segment_duration=15, output_dir=output_path).set_signal_processors(sp.STFT, sp.MEL_SPEC, sp.CQT)

    # preprocess
    if args.process:
        preprocessor.preprocess()

    # create examples
    if args.figures:
        pu.create_graph_example_figures(sp.STFT, sp.MEL_SPEC, sp.CQT, song_paths=preprocessor.get_songs(), figures_path=figures_path, num_songs=args.figures)
