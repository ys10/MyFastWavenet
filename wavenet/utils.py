import numpy as np
import wave

from scipy.io import wavfile


def normalize(data):
    temp = np.float32(data) - np.min(data)
    out = (temp / np.max(temp) - 0.5) * 2
    return out


def make_batch(path):
    data = wavfile.read(path)[1][:, 0]

    data_ = normalize(data)
    # data_f = np.sign(data_) * (np.log(1 + 255*np.abs(data_)) / np.log(1 + 255))

    bins = np.linspace(-1, 1, 256)
    # Quantize inputs.
    inputs = np.digitize(data_[0:-1], bins, right=False) - 1
    inputs = bins[inputs][None, :, None]

    # Encode targets as ints.
    targets = (np.digitize(data_[1::], bins, right=False) - 1)[None, :]
    return inputs, targets


def write2wave(path, data, nframes, nchannels=1, sampwidth=2, framerate=44100):
    with wave.open(path, "w")as wave_file:
        wave_file.setframerate(framerate)
        wave_file.setnchannels(nchannels)
        wave_file.setnframes(nframes)
        wave_file.setsampwidth(sampwidth)
        wave_file.writeframes(data)
        pass
    pass
