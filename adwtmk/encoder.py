import numpy as np
from adwtmk.audio import Audio
from adwtmk.utilities import *


class WaterMarkEncodeError(Exception):
    pass


class MarkTooLargeError(WaterMarkEncodeError):
    pass


class MarkFormatError(WaterMarkEncodeError):
    pass


def lsb_encode(original_audio: Audio, mark: bytes)->Audio:
    """
    用LSB对音频进行隐写，返回新的Audio对象，同时在Audio.key中保存解码所用的key。

    :param original_audio: 原音频，为一个Audio对象
    :param mark: 要隐写的内容，为一个bytes对象
    :return: 隐写后的音频，为一个Audio对象
    """
    if not isinstance(mark, bytes):
        raise MarkFormatError("Mark must be a bytes object.")
    original_samples = original_audio.get_array_of_samples()
    samples_len = len(original_samples)
    if samples_len < 8*len(mark):
        raise MarkTooLargeError("Mark too large for LSB encoding.")
    low_bits = get_all_bits(mark)
    bits_len = len(low_bits)
    assert(8*len(mark) == bits_len)
    key = np.random.randint(0, samples_len, size=bits_len)
    marked_samples = original_samples
    for i in range(bits_len):
        marked_samples[key[i]] = (marked_samples[key[i]] & -2) + low_bits[i] # maybe better solution?
    new_audio = original_audio.spawn(marked_samples)
    assert(len(marked_samples) == len(original_samples))
    new_audio.key = {'type': "LSB", 'key': key.tolist()}
    return new_audio
