from adwtmk.audio import Audio
from functools import reduce
from array import array


class WaterMarkDecodeError(Exception):
    pass


class WrongKeyError(WaterMarkDecodeError):
    pass


class KeyNotFoundError(WaterMarkDecodeError):
    pass


def dectect_key(key_type: str):
    def detect(func):
        def wrapper(marked_audio: Audio, key: dict)->bytes:
            if marked_audio.key is None and key is None:
                raise KeyNotFoundError("No key found.")
            if marked_audio.key is None or marked_audio.key.get("type", None) != key_type and \
               key is None or key.get("type", None) != key_type:
                raise WrongKeyError("Wrong type of key.")
            if marked_audio.key.get("type", None) == key_type:
                key = marked_audio.key
            if key.get('key', None) is None:
                raise KeyNotFoundError("No key found.")
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
    key_list = key['key']
    if not isinstance(key_list, list):
        raise WrongKeyError("Wrong type of key.")
    if len(key_list) % 8 != 0:
        raise WrongKeyError("Wrong length of key.")
    samples = marked_audio.get_array_of_samples()
    decoded_bytes = array('B', [])
    try:
        low_bits = [samples[index] & 1 for index in key_list]
        for i in range(len(key_list)//8):
            decoded_bytes.append(int(reduce(lambda x, y: (x | y) << 1, low_bits[8*i:8*i+8])>>1))
    except IndexError:
        raise WrongKeyError("Invalid key.")
    return decoded_bytes.tobytes()
