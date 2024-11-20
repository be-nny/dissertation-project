import os

from preprocessor import preprocessor as p
from preprocessor import signal_processor as sp
from dotenv import load_dotenv

dotenv_path = "config.env"
load_dotenv(dotenv_path=dotenv_path)
dataset_path = os.getenv("DATASET_PATH")

preprocessor = p.Preprocessor(dataset_dir=dataset_path, segment_duration=15).set_signal_filters(sp.STFT, sp.MEL_SPEC, sp.CQT).preprocess()
