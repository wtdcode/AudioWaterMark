import numpy as np
from adwtmk.audio import Audio


class WaterMarkEncodeError(Exception):
    pass


class MarkTooLargeError(WaterMarkEncodeError):
    pass


def lsb_encode(original_audio: Audio, mark: bytes)->Audio:
    original_samples = original_audio.get_array_of_samples()
    samples_len = len(original_samples)
    if samples_len < 8*len(mark):
        raise MarkTooLargeError("Mark too large for LSB encoding.")
    low_bits = [bit for L in map(lambda byte: [((byte & (1 << i)) >> i) for i in range(7, -1, -1)], mark) for bit in L]
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
