from file import MidiFile
import glob
import numpy as np
from scipy import sparse
from random import shuffle

def save_piano_rolls(file_path):

    f = glob.glob(file_path)
    mf = MidiFile.from_mid(f)
    mf.save_piano_rolls()

def get_piano_rolls(file_path, size_per_channel = 128*3, num_channels = 1):

    f = glob.glob(file_path)
    mf = MidiFile.from_npz(f)

    data_arr = []

    for mf_arr in mf._arrays:
        r, c = mf_arr.shape
        size = size_per_channel * num_channels
        dc = (size - c % size) % size
        arr = np.zeros((r, c + dc))
        arr[:,0:c] = mf_arr.toarray()[:,0:c]
        for i in range(0, c + dc, size):
            channel_arr = np.zeros((num_channels, r, size_per_channel))
            for cx, j in enumerate(range(i, i + size, size_per_channel)):
                channel_arr[cx] = arr[:,j : j + size_per_channel]
            data_arr.append(channel_arr)

    return data_arr

class DataGen(object):

    def __init__(self, file_path, batch_size, num_channels, width = 128*3, seed=4):
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.width = width
        self.seed = seed
        np.random.seed(seed=seed)
        self._data = get_piano_rolls(file_path, width, num_channels)

    def __len__(self):
        return len(self._data)

    def gen_batch(self):
        indx = np.random.randint(0, high = len(self._data), size = self.batch_size)
        data = [self._data[i] for i in indx]
        return np.stack(data)

if __name__ == '__main__':
    data = DataGen('data/adam/*.npz', 68, 9)
    print("loaded {0} datasets".format(len(data)))
    print(data.gen_batch().shape)
