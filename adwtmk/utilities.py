from array import array
from functools import reduce
from pydub.utils import ARRAY_RANGES
import numpy as np


def get_all_bits(byte_string: bytes, reverse: bool=False)->list:
    if not reverse:
        return [bit
                for L in map(lambda byte: [((byte & (1 << i)) >> i) for i in range(7, -1, -1)], byte_string)
                for bit in L]
    else:
        return [bit
                for L in map(lambda byte: [((byte & (1 << i)) >> i) for i in range(8)], byte_string)
                for bit in L]


def get_all_bytes(byte_list: list)->bytes:
    decoded_bytes = array('B', [])
    bytes_len = len(byte_list) // 8
    for i in range(bytes_len):
        decoded_bytes.append(int(reduce(lambda x, y: (x | y) << 1, byte_list[8 * i:8 * i + 8]) >> 1))
    return decoded_bytes.tobytes()


def reg_samples(samples: np.array, sample_width)->np.array:
    max_sample = ARRAY_RANGES[sample_width*8][1]
    return samples / max_sample


def stretch_samples(samples: np.array, sample_width)->np.array:
    samples[samples > 1.0] = 1.0
    samples[samples < -1.0] = -1.0
    max_sample = ARRAY_RANGES[sample_width*8][1]
    return samples * max_sample
