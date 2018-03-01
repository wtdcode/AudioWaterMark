from adwtmk.encoder import *
from adwtmk.decoder import *

# read sound file
sound = Audio.from_file("./mark.flac", format="flac")

# basic information
print(sound.frame_rate, sound.duration_seconds, len(sound.get_array_of_samples()))

# metadata
print(sound.tags)

# same as audioread in matlab
print(sound.get_reshaped_samples())

# LSB encoding
LSB_marked = lsb_encode(sound, b"LSB")

# LSB decoding
print(lsb_decode(LSB_marked))

# echo encoding
ECHO_marked = echo_encode(sound, b'ECHO')

# echo decoding
print(echo_decode(ECHO_marked))

# spawn a new audio like audiowrite in matlab
samples = ECHO_marked.get_reshaped_samples()
new_audio = ECHO_marked.spawn(samples)

# and export it with key
new_audio.export_with_key(key_path="./test/key.json", out_path="./test/mark.flac", format="flac", tags=ECHO_marked.tags)
