import argparse
import yaml

from mat_preprocessor import preprocessor as p
from mat_preprocessor import utils as pu
from mat_preprocessor.signal_processor import get_type, get_all_types
from jsonschema import validate
from mat_logger import mat_logger
import mat_config
# arguments parser
parser = argparse.ArgumentParser(prog='Music Analysis Tool (MAT) - PREPROCESSOR', formatter_class=argparse.RawDescriptionHelpFormatter, description="Preprocess Audio Dataset")
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-s", "--signal_processors", required=True, choices=get_all_types(), nargs="+", help="the signal processors to apply to the raw audio")
parser.add_argument("-p", "--process", action="store_true", help="preprocesses data the data use the parameters set in the config file")
parser.add_argument("-f", "--figures", action="store", default=1, type=int, help="create a set of n example figures")

if __name__ == "__main__":
    args = parser.parse_args()
    logger = mat_logger.get_logger()

    # load config file
    if args.config:
        config = mat_config.Config(path=args.config)

    # get signal processor args
    signal_processors = []
    if args.signal_processors:
        for s in args.signal_processors:
            try:
                name = get_type(s)
                signal_processors.append(get_type(s))
            except ValueError as e:
                logger.error(e)
                raise e

    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=config.DATASET_PATH, target_length=config.TARGET_LENGTH, segment_duration=config.SEGMENT_DURATION, output_dir=config.OUTPUT_PATH, logger=logger, train_split=config.TRAIN_SPLIT).set_signal_processors(*signal_processors)

    # create examples
    if args.figures:
        pu.create_graph_example_figures(*preprocessor.get_signal_processors(), song_paths=preprocessor.get_songs(), figures_path=preprocessor.get_figures_path(), num_songs=args.figures)

    # preprocess
    if args.process:
        preprocessor.preprocess()
