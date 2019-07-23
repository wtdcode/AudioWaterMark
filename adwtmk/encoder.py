import numpy.matlib as mt
from adwtmk.audio import Audio
from adwtmk.utilities import *
from scipy.signal import fftconvolve
import pyfftw


class WaterMarkEncodeError(Exception):
    pass


class MarkTooLargeError(WaterMarkEncodeError):
    pass


class MarkFormatError(WaterMarkEncodeError):
    pass


def lsb_encode(original_audio: Audio, mark: bytes) -> Audio:
    """
    用LSB对音频进行隐写。

    优点：隐写内容多，实现简单。
    缺点：鲁棒性差，隐写数据易失。

    :param original_audio: 原音频，为一个Audio对象
    :param mark: 要隐写的内容，为一个bytes对象
    :return: 隐写后的音频，为一个Audio对象
    """
    if not isinstance(mark, bytes):
        raise MarkFormatError("Mark must be a bytes object.")
    original_samples = original_audio.get_array_of_samples()
    samples_len = len(original_samples)
    if samples_len < 8 * len(mark):
        raise MarkTooLargeError("Mark too large for LSB encoding.")
    low_bits = get_all_bits(mark)
    bits_len = len(low_bits)
    assert (8 * len(mark) == bits_len)
    key = np.random.choice(samples_len, bits_len, replace=False)
    marked_samples = original_samples
    for i in range(bits_len):
        marked_samples[key[i]] = (marked_samples[key[i]] & -2) + low_bits[i]  # 这里其实可以用异或来做，简单起见这里先抹掉了最低位然后加上要隐写的内容
    new_audio = original_audio.spawn(marked_samples)
    assert (len(marked_samples) == len(original_samples))
    new_audio.key = {'type': "LSB", 'key': key.tolist()}
    return new_audio


def echo_encode(original_audio: Audio, mark: bytes, alpha: float = 0.7, m: tuple = (150, 200)) -> Audio:
    """
    用回声隐藏进行隐写。

    优点：透明性强，写入的数据并非噪声，鲁棒性好，抗压缩性好。
    缺点：容量小，这里默认是8192个取样点写入一个比特。

    :param original_audio: 原音频，为一个Audio对象
    :param mark: 水印，必须为一个bytes对象
    :param alpha: 衰退率，理论上这个值越大水印鲁棒性越好，但生成的音频回音会更强
    :param m: 回音的延迟，分别为比特0和比特1的延迟
    :return: 隐写后的音频，为一个Audio对象
    """
    if not isinstance(mark, bytes):
        raise MarkFormatError("Mark must be a bytes object.")
    if alpha > 1 or alpha < 0:
        raise MarkFormatError("Alpha must be in [0,1].")
    bits = np.mat(get_all_bits(mark))
    bits_len = bits.shape[1]
    original_samples_reg = np.matrix(original_audio.get_reshaped_samples())
    samples_len = original_samples_reg.shape[1]
    channels = original_audio.channels
    if 8192 * bits_len > samples_len:
        raise MarkTooLargeError("Mark too large for echo encoding.")
    fragment_len = samples_len // bits_len
    encoded_len = bits_len * fragment_len
    kernel0 = np.concatenate((mt.zeros((channels, m[0])), alpha * original_samples_reg), 1)
    kernel1 = np.concatenate((mt.zeros((channels, m[1])), alpha * original_samples_reg), 1) # 这里是两个回声核，0代表隐藏信息的0，1代表隐藏信息的1
    direct_sig = np.reshape(np.matrix(mt.ones((fragment_len, 1))) * bits, (encoded_len, 1), 'F')
    smooth_length = int(np.floor(fragment_len/4) - np.floor(fragment_len/4) % 4)
    tp = np.matrix(fftconvolve(np.array(direct_sig)[:, 0], np.hanning(smooth_length))) # 这里用的是汉宁窗口，其他窗口的效果类似
    window = tp[0, smooth_length // 2:tp.shape[1] - smooth_length // 2 + 1] / np.max(np.abs(tp)) # window 用于平滑
    mixer = mt.ones((channels, 1)) * window # 这里的 channels 很关键，需要写入所有的音轨
    encoded_samples_reg = original_samples_reg[:, :encoded_len] + \
                          np.multiply(kernel1[:, :encoded_len], mixer) + \
                          np.multiply(kernel0[:, :encoded_len], abs(mixer - 1)) # 这里是回音合成
    new_samples_reg = np.concatenate((encoded_samples_reg, original_samples_reg[:, encoded_len:]), 1)
    new_audio = original_audio.spawn(new_samples_reg)
    new_audio.key = {'type': 'SINGLE_ECHO',
                     'key': {
                         'm': m,
                         'fragment_len': fragment_len,
                         'bits_len': bits_len
                     }}
    return new_audio


def dft_encode(original_audio: Audio, mark: bytes, alpha: float = 0.5, smooth: bool = True):
    """
    用频域进行隐写。

    优点：抗压缩性强，容量较大
    缺点：有一定失真

    :param original_audio: 原音频，为一个Audio对象
    :param mark: 水印，必须为一个bytes对象
    :param alpha: 衰退率，理论上这个值越大水印鲁棒性越好，但是杂音会更加严重
    :param smooth: 考虑到浮点的误差对隐写后的音频进行平滑
    :return: 隐写后的音频，为一个Audio对象
    """
    # 这里是当初一个大坑
    # numpy 默认的 FFT 算法是一个开源算法，跑一次要几个小时
    # FFTW 是一个针对固定大小 FFT 优化的算法，非常适合我们的场景，替换后每次只要几分钟
    np.fft.fft = pyfftw.interfaces.numpy_fft.fft
    np.fft.ifft = pyfftw.interfaces.numpy_fft.ifft
    if not isinstance(mark, bytes):
        raise MarkFormatError("Mark must be a bytes object.")
    bits = np.array(get_all_bits(mark))
    bits_len = len(bits)
    original_samples_reg = original_audio.get_array_of_regular_samples()
    samples_len = len(original_samples_reg)
    if 2*bits_len > samples_len:
        raise MarkTooLargeError("Mark too large for DFT encoding.")
    bits[bits == 0] = -1
    original_spectrum = np.fft.fft(original_samples_reg, planner_effort="FFTW_ESTIMATE")
    random_key = np.random.choice((samples_len-1)//2, bits_len, replace=False) + 1
    marked_spectrum = original_spectrum
    # 这里很关键一点是要对称！
    # 因为 DFT 是对称的，只有对称添加才能减少失真
    # 另外一点，这里在添加的时候完全随机化实际上导致了一些失真，其实有一些其他的统计特征可以帮助选择添加水印的区域
    marked_spectrum[random_key] += (bits * alpha)
    marked_spectrum[samples_len - random_key] += (bits*alpha)
    marked_samples_reg = np.fft.ifft(marked_spectrum, planner_effort="FFTW_ESTIMATE")
    # 原音频为空白的地方平滑为空白
    if smooth:
        marked_samples_reg[original_samples_reg == 0.0] = 0.0
    new_audio = original_audio.spawn(np.real(marked_samples_reg))
    new_audio.key = {'type': 'DFT',
                     'key': {
                         'random_key': random_key.tolist()
                     }}
    return new_audio
