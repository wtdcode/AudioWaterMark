from pydub import AudioSegment
from pydub.utils import get_array_type
from adwtmk.utilities import *
import numpy as np
import mutagen
import json
from array import array

SAMPLE_ACCURACY = {1: np.int8,
                   2: np.int16,
                   4: np.int32,
                   8: np.int64}


class Audio(AudioSegment):
    def __init__(self, data=None, tags=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.tags = tags
        self.key = None

    @classmethod
    def from_file(cls, file: str, format=None, codec=None, parameters=None, **kwargs):
        obj = super().from_file(file, format, codec, parameters, **kwargs)
        tag = mutagen.File(file)
        obj.tags = {}
        for k, v in tag.items():
            obj.tags[k] = v[0]
        return obj

    def get_reshaped_samples(self)->np.ndarray:
        samples = self.get_array_of_samples()
        channels = self.channels
        channel_length = len(samples) // channels
        return reg_samples(np.reshape(np.array(samples), (channels, channel_length), 'F').copy(), self.sample_width)

    def spawn(self, data, overrides: dict = {}): # just make pydub and PEP8 happy :P
        if isinstance(data, list):
            data = array(get_array_type(self.sample_width*8), data)
        if isinstance(data, np.ndarray):
            data = Audio.get_flatten_samples(data)
            data = stretch_samples(data, self.sample_width).tolist()
            data = array(get_array_type(self.sample_width*8), data)
        return self._spawn(data, overrides)

    def _spawn(self, data: list, overrides: dict={}):
        new_audio = super()._spawn(data, overrides)
        new_audio.tags = self.tags
        new_audio.key = self.key
        return new_audio

    @staticmethod
    def get_flatten_samples(arr: np.ndarray)->np.ndarray:
        return np.array(arr).flatten('F').copy()

    def export(self, out_f=None, format='mp3', codec=None, bitrate=None, parameters=None, tags=None, id3v2_version='4', cover=None)->None:
        out_fp = super().export(out_f, format, codec, bitrate, parameters, tags, id3v2_version, cover)
        out_fp.close()
        return

    def export_with_key(self, key_path, out_path=None, format='mp3', codec=None, bitrate=None, parameters=None, tags=None, id3v2_version='4', cover=None)->None:
        super().export(out_path, format, codec, bitrate, parameters, tags, id3v2_version, cover)
        with open(key_path, mode="w+") as f:
            json.dump(self.key, f, indent=4)
        return

