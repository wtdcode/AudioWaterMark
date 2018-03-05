from adwtmk.encoder import *
from adwtmk.decoder import *
from pydub.utils import get_array_type

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
def audiowrite(samples: array,
               frame_rate: int, sample_width: int, channels: int,
               key_path: str, out_path: str, format: str,
               tags: dict)->None:
    tp = Audio.silent(len(samples)/(channels*frame_rate) * 1000, frame_rate)
    tp = tp.spawn(samples, {'sample_width': sample_width,
                            'frame_rate': frame_rate,
                            'channels': channels})
    tp.export_with_key(key_path=key_path, out_path=out_path, format=format, tags=tags)
    return


# an implementation more OOP
def audiowrite_OOP(audio: Audio, key_path: str, out_path: str, format: str)->None:
    samples = audio.get_array_of_samples()
    new_audio = audio.spawn(samples)
    new_audio.export_with_key(key_path=key_path, out_path=out_path, format=format, tags=audio.tags)
    return


audiowrite(LSB_marked.get_array_of_samples(),
           LSB_marked.frame_rate,
           LSB_marked.sample_width,
           LSB_marked.channels,
           "./test/LSB.json", "./test/LSB_marked.flac", "flac",
           LSB_marked.tags)
audiowrite(ECHO_marked.get_array_of_samples(),
           ECHO_marked.frame_rate,
           ECHO_marked.sample_width,
           ECHO_marked.channels,
           "./test/ECHO,json", "./test/ECHO_marked.flac", "flac",
           ECHO_marked.tags)
audiowrite(DFT_marked.get_array_of_samples(),
           DFT_marked.frame_rate,
           DFT_marked.sample_width,
           DFT_marked.channels,
           "./test/DFT.json", "./test/DFT_marked.flac", "flac",
           DFT_marked.tags)


# and we can embed an audio as water mark
# encoding
mark = Audio.from_file("./mark.flac", format="flac")
DFT_marked_with_audio = dft_encode(sound, mark[:5000].raw_data, smooth=True)
DFT_marked_with_audio.key['key']['metadata'] = {
    'sample_width': mark.sample_width,
    'frame_rate': mark.frame_rate,
    'channels': mark.channels
}
DFT_marked_with_audio.key['key']['tags'] = mark.tags

# decoding
decoded_bytes = dft_decode(DFT_marked_with_audio, {'type': 'DFT',
                                                   'key': {
                                                      'random_key': DFT_marked_with_audio.key['key']['random_key'],
                                                      'original_audio': sound
                                                    }})
mark_sample_width = DFT_marked_with_audio.key['key']['metadata']['sample_width']
mark_frame_rate = DFT_marked_with_audio.key['key']['metadata']['frame_rate']
mark_channels = DFT_marked_with_audio.key['key']['metadata']['channels']
mark_tags = mark.tags
decoded_samples = array(get_array_type(mark_sample_width*8), decoded_bytes) # should be packaged into Audio
audiowrite(decoded_samples, mark_frame_rate, mark_sample_width, mark_channels,
           "./test/DFT_AUDIO.json", "./test/extracted.flac", "flac", mark_tags)

# calculate BER
from adwtmk.utilities import get_all_bits

total_bits = get_all_bits(decoded_bytes)

total_bits_len = len(total_bits)

BER = np.sum(np.array(np.array(total_bits) - np.array(get_all_bits(mark[:5000].raw_data))) != 0) / total_bits_len

print(BER)
