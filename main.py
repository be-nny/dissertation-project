import os

from preprocessor import preprocessor as p
from preprocessor import signal_processor as sp
from preprocessor import utils as pu
from dotenv import load_dotenv

dotenv_path = "config.env"
load_dotenv(dotenv_path=dotenv_path)
dataset_path = os.getenv("DATASET_PATH")
output_path = os.getenv("OUTPUT_PATH")

preprocessor = p.Preprocessor(dataset_dir=dataset_path, segment_duration=15, output_dir=output_path).set_signal_processors(sp.STFT, sp.MEL_SPEC, sp.CQT)
preprocessor.preprocess()

# pu.create_graph_example_figures(sp.STFT, sp.MEL_SPEC, sp.CQT, song_paths=preprocessor.get_songs(), num_songs=2)
