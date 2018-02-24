from pydub import AudioSegment
from adwtmk.audio import Audio
from adwtmk.encoder import *
from adwtmk.decoder import *

sound1 = Audio.from_file("./mark.flac", format="flac")

print(sound1.frame_rate,sound1.duration_seconds, len(sound1.get_array_of_samples()))

print(sound1.get_reshaped_samples())

marked = lsb_encode(sound1, b"woccccccc")

marked.export_with_key(key_path="./test/key.json", out_path="./test/mark.flac", format="flac", tags=marked.tags)

print(lsb_decode(marked))