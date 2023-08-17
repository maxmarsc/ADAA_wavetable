import numpy as np
import soundfile as sf
import soxr
from pathlib import Path

from math import log2
from enum import Enum

from numba import njit

class Waveform:
    def __init__(self, data: np.ndarray, samplerate: int):
        # data, sr = sf.read(path, dtype=np.float32)
        self.__data = data
        self.__sr = samplerate
    
    @property
    def size(self) -> int:
        return self.__data.shape[0]
    
    def get(self, size: int = None) -> np.ndarray[float]:
        if size == None or size == self.size:
            return self.__data[:]
        else:
            ratio = float(size / self.size)
            o_sr = self.__sr * ratio
            # return self.__resample(ratio)
            resampled = soxr.resample(
                self.__data,
                self.size,
                size,
                quality="VHQ"
            )
            return resampled / np.abs(np.max(resampled))
        
    def __resample(self, ratio: float) -> np.ndarray[float]:
        triple = np.zeros(self.size * 3)
        o_sr = self.__sr * ratio
        o_size = int(np.floor(self.size * ratio))
        for i in range(3):
            triple[self.size*i: self.size*(i+1)] = self.__data

        resampled = soxr.resample(
            triple,
            self.__sr,
            o_sr,
            quality="VHQ"
        )

        return resampled[o_size: o_size*2]

        
class FileWaveform(Waveform):
    def __init__(self, path: Path):
        data, sr = sf.read(path, dtype=np.float32)
        super().__init__(data, sr)


class NaiveWaveform(Waveform):
    class Type(Enum):
        SAW = 0
        SQUARE = 1
        TRIANGLE = 2
        SIN = 3
    
    def __init__(self, type: Type, size: int, samplerate: int):
        if type == NaiveWaveform.Type.SAW:
            data = NaiveWaveform.__compute_saw(size)
        elif type == NaiveWaveform.Type.SIN:
            data = NaiveWaveform.__compute_sin(size)
        elif type ==NaiveWaveform.Type.SQUARE:
            data = NaiveWaveform.__compute_square(size)
        else:
            raise NotImplementedError
        
        super().__init__(data, samplerate)

    @staticmethod
    @njit
    def __compute_saw(size: int) -> np.ndarray:
        phase = 0.0
        waveform = np.zeros(size)
        step = 1.0/size

        for i in range(size):
            waveform[i] = 2.0 * phase - 1
            phase = (phase + step) % 1.0

        return waveform
    
    @staticmethod
    @njit
    def __compute_sin(size: int) -> np.ndarray:
        phase = np.linspace(0, 2*np.pi, size + 1)
        return np.sin(phase[:-1])
    
    @staticmethod
    @njit
    def __compute_square(size: int) -> np.ndarray:
        data = np.ones(size)
        data[size/2:] *= -1.0
        return data


    
if __name__ == "__main__":
    wave = NaiveWaveform(NaiveWaveform.Type.SAW, 2048, 44100)

    sizes = (2048, 512, 256, 128, 64, 32)

    waves = [wave.get(size) for size in sizes]

    # wave_og = wave.get()
    # wave_1024 = wave.get(1024)
    # wave_512 = wave.get(512)
    # wave_256 = wave.get(256)
    # wave_128 = wave.get(128)
    # wave_64 = wave.get(64)

    # import soundfile as sf
    # sf.write("naive_saw.wav", wave_og, 44100)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(nrows=3, ncols=2)

    for i, ax in enumerate(axs.flat):
        waveform = waves[i]
        print("{} : {}".format(waveform.shape[0], np.max(np.abs(waveform))))
        ax.plot(waveform, label="{}".format(waveform.shape[0]))
        ax.legend()
    # axs[0].plot(wave_og, label="2048")
    # axs[1].plot(wave_1024, label="1024")
    # axs[2].plot(wave_512, label="512")
    # axs[3].plot(wave_256, label="256")
    # axs[4].plot(wave_128, label="128")
    # axs[5].plot(wave_64, label="64")

    plt.show()