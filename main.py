import os
import argparse

from preprocessor import preprocessor as p
from preprocessor import signal_processor as sp
from preprocessor import utils as pu
from dotenv import load_dotenv

# arguments parser
parser = argparse.ArgumentParser(description="Music Analysis Tool")
parser.add_argument("-p", "--process", action="store_true", help="Preprocess data")
parser.add_argument("-c", "--config", required=True, help="Config file")
parser.add_argument("-e", "--examples", default=1, type=int, help="Create example figures")

if __name__ == "__main__":
    args = parser.parse_args()

    # load config file
    if args.config:
        load_dotenv(dotenv_path=args.config)
        dataset_path = os.getenv("DATASET_PATH")
        output_path = os.getenv("OUTPUT_PATH")
        figures_path = os.getenv("FIGURES_PATH")

    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=dataset_path, segment_duration=15, output_dir=output_path).set_signal_processors(sp.STFT, sp.MEL_SPEC, sp.CQT)

    # preprocess
    if args.process:
        preprocessor.preprocess()

    # create examples
    if args.examples:
        pu.create_graph_example_figures(sp.STFT, sp.MEL_SPEC, sp.CQT, song_paths=preprocessor.get_songs(), figures_path=figures_path, num_songs=args.examples)
