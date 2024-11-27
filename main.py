import argparse
import yaml

from preprocessor import preprocessor as p
from preprocessor import utils as pu
from preprocessor.signal_processor import get_type, get_all_types
from jsonschema import validate

# arguments parser
parser = argparse.ArgumentParser(prog='DATA PREPROCESSOR',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="""
Preprocess the audio dataset
         
'config.env' FIELDS
-------------------
DATASET_PATH=''
PREPROCESSED_PATH=''
FIGURES_PATH=''
""")
parser.add_argument("-c", "--config", required=True, help="config file")
parser.add_argument("-s", "--signal_processors", required=True, choices=get_all_types(), nargs="+", help="the signal processors to apply to the audio")
parser.add_argument("-p", "--process", action="store_true", help="preprocesses data the data use the parameters set in the config file")
parser.add_argument("-f", "--figures", action="store", default=1, type=int, help="create a set of n example figures")

if __name__ == "__main__":
    args = parser.parse_args(["--config=config.yml", "--signal_processors=STFT", "-f 1"])

    # load config file
    if args.config:
        with open(args.config, 'r') as f:
            yml_data = yaml.load(f, Loader=yaml.FullLoader)

        # validating schema
        with open("schema.yml", 'r') as schema:
            validate(yml_data, yaml.load(schema, Loader=yaml.FullLoader))

        dataset_path = yml_data["dataset"]
        output_path = yml_data["preprocessor_config"]["preprocessed_output"]
        figures_path = yml_data["preprocessor_config"]["figures_output"]
        target_length = yml_data["preprocessor_config"]["target_length"]
        segment_duration = yml_data["preprocessor_config"]["segment_duration"]

    # get signal processor args
    signal_processors = []
    if args.signal_processors:
        for s in args.signal_processors:
            try:
                name = get_type(s)
                signal_processors.append(get_type(s))
            except ValueError as e:
                raise e

    # creating a preprocessor
    preprocessor = p.Preprocessor(dataset_dir=dataset_path, target_length=target_length, segment_duration=segment_duration, output_dir=output_path).set_signal_processors(*signal_processors)

    # preprocess
    if args.process:
        preprocessor.preprocess()

    # create examples
    if args.figures:
        pu.create_graph_example_figures(*preprocessor.get_signal_processors(), song_paths=preprocessor.get_songs(), figures_path=figures_path, num_songs=args.figures)
