import argparse
import config
from preprocessor import preprocessor as p
from preprocessor import utils as pu
from preprocessor.signal_processor import get_type, get_all_types
import logger

# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - PREPROCESSOR', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-s", "--signal_processor", required=True, choices=get_all_types(), help="the signal processor to apply to the raw audio")
parser.add_argument("-f", "--figures", action="store", default=1, type=int, help="create a set of n example figures")

if __name__ == "__main__":
    args = parser.parse_args()
    logger = logger.get_logger()

    # load config file
    config_obj = config.Config(path=args.config)

    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=config_obj.DATASET_PATH, target_length=config_obj.TARGET_LENGTH, segment_duration=config_obj.SEGMENT_DURATION, sample_rate=config_obj.SAMPLE_RATE, output_dir=config_obj.OUTPUT_PATH, logger=logger, train_split=config_obj.TRAIN_SPLIT).set_signal_processor(get_type(args.signal_processor))

    # create examples
    if args.figures:
        pu.create_graph_example_figures(preprocessor.get_signal_processor(), song_paths=preprocessor.get_songs(), figures_path=preprocessor.get_figures_path(), num_songs=args.figures)

    # start preprocessing
    preprocessor.preprocess()
