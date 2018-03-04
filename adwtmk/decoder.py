from adwtmk.audio import Audio
from adwtmk.utilities import *
import numpy as np
import pyfftw

class WaterMarkDecodeError(Exception):
    pass


class WrongKeyError(WaterMarkDecodeError):
    pass


class KeyNotFoundError(WaterMarkDecodeError):
    pass


def dectect_key(key_type: str):
    def detect(func):
        def wrapper(marked_audio: Audio, key: dict=None)->bytes:
            if marked_audio.key is None and key is None:
                raise KeyNotFoundError("No key found.")
            if (marked_audio.key is None or marked_audio.key.get("type", None) != key_type) and \
               (key is None or key.get("type", None) != key_type):
                raise WrongKeyError("Wrong type of key.")
            if key is None or key.get("type", None) != key_type:
                key = marked_audio.key
            if key.get('key', None) is None:
                raise KeyNotFoundError("No key found.")
            key = key['key']
            return func(marked_audio, key)
        return wrapper
    return detect


@dectect_key("LSB")
def lsb_decode(marked_audio: Audio, key: dict=None)->bytes:
    """
    用Audio.key或者传入的key对音频解LSB码，返回解码所得的bytes。

    :param marked_audio: 用LSB隐写的音频。
    :param key: （可选）解码用的key，**如果Audio.key是有效的，这个参数会被忽略。**
    :return: 隐写的bytes。
    """
    key_list = key
    if not isinstance(key_list, list):
        raise WrongKeyError("Wrong type of key.")
    if len(key_list) % 8 != 0:
        raise WrongKeyError("Wrong length of key.")
    samples = marked_audio.get_array_of_samples()
    low_bits = [samples[index] & 1 for index in key_list]
    decoded_bytes = None
    try:
        decoded_bytes = get_all_bytes(low_bits)
    except IndexError:
        raise WrongKeyError("Invalid key.")
    return decoded_bytes


@dectect_key("SINGLE_ECHO")
def echo_decode(marked_audio: Audio, key: dict=None)->bytes:
    np.fft.fft = pyfftw.interfaces.numpy_fft.fft
    np.fft.ifft = pyfftw.interfaces.numpy_fft.ifft
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    m = key['m']
    fragment_len = key['fragment_len']
    bits_len = key['bits_len']
    encoded_len = fragment_len*bits_len
    samples_width = marked_audio.sample_width
    marked_samples_reg = marked_audio.get_reshaped_samples()
    channel0_samples_reg = np.reshape(marked_samples_reg[0, :encoded_len], (fragment_len, bits_len), 'F')
    bits = []
    for i in range(bits_len):
        rcep = np.real(
            np.fft.ifft(
                np.log(
                    np.abs(
                        np.fft.fft(
                            np.multiply(
                                channel0_samples_reg[:, i],
                                np.bartlett(fragment_len)
                            ))))))
        if rcep[m[0]] >= rcep[m[1]]:
            bits.append(0)
        else:
            bits.append(1)
    try:
        decoded_bytes = get_all_bytes(bits)
    except IndexError:
        raise WrongKeyError("Invalid key.")
    return decoded_bytes
