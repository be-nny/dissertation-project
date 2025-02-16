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

# parser.add_argument("-n", "--n_fft", action="store", type=int)
# parser.add_argument("-hl", "--hop_length", action="store", type=int)
# parser.add_argument("-nm", "--n_mels", action="store", type=int)
# parser.add_argument("-nps", "--nperseg", action="store", type=int)
# parser.add_argument("-sr", "--scale_range", required=True, type=tuple, help="range to scale between")

if __name__ == "__main__":
    args = parser.parse_args()
    logger = logger.get_logger()

    # load config file
    if args.config:
        config = config.Config(path=args.config)

    # set signal processing vars


    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=config.DATASET_PATH, target_length=config.TARGET_LENGTH, segment_duration=config.SEGMENT_DURATION, output_dir=config.OUTPUT_PATH, logger=logger, train_split=config.TRAIN_SPLIT).set_signal_processor(get_type(args.signal_processor))

    # create examples
    if args.figures:
        pu.create_graph_example_figures(preprocessor.get_signal_processor(), song_paths=preprocessor.get_songs(), figures_path=preprocessor.get_figures_path(), num_songs=args.figures)

    # start preprocessing
    preprocessor.preprocess()
