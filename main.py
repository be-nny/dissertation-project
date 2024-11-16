from preprocessor import preprocessor as p
from preprocessor import signal_processor as sp
from preprocessor import utils

preprocessor = p.Preprocessor(segment_duration=30).set_signal_filters(sp.STFT, sp.MEL_SPEC)

path = "test_files/audio_2.mp3"
print(utils.get_song_metadata(path=path))

preprocessor.process(path=path)
