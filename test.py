from adwtmk.encoder import *
from adwtmk.decoder import *

# read sound file
sound = Audio.from_file("./test.flac", format="flac")

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

# DFT encoding (API needs to be optimized next version)
DFT_marked = dft_encode(sound, b'DFT_ENCODING_CAN_CONTAIN_VERY_LONG_MARK')
# DFT decoding needs original audio!
DFT_key = {'type': 'DFT',
           'key': {
              'random_key': DFT_marked.key['key']['random_key'],
              'original_audio': sound
            }}
print(dft_decode(DFT_marked, DFT_key))


# spawn a new audio like audiowrite in matlab
def audiowrite(audio: Audio, key_path: str, out_path: str, format: str)->None:
    samples = audio.get_array_of_samples()
    new_audio = audio.spawn(samples)
    new_audio.export_with_key(key_path=key_path, out_path=out_path, format=format, tags=audio.tags)
    return

audiowrite(LSB_marked, "./test/LSB.json", "./test/LSB_marked.flac", "flac")
audiowrite(ECHO_marked, "./test/ECHO,json", "./test/ECHO_marked.flac", "flac")
audiowrite(DFT_marked, "./test/DFT.json", "./test/DFT_marked.flac", "flac")
