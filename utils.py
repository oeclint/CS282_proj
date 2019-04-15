from file import MidiFile
import glob
import numpy as np
from scipy import sparse
from random import shuffle

def save_piano_rolls(file_path):

    f = glob.glob(file_path)
    mf = MidiFile.from_mid(f)
    mf.save_piano_rolls()

def get_piano_rolls(file_path, size = 128*3, random = True):

    f = glob.glob(file_path)
    mf = MidiFile.from_npz(f)

    arr_part = []

    for arr in mf._arrays:
        r, c = arr.shape
        dc = size - c % size
        exp_arr = np.zeros((r, c + dc))
        exp_arr[:,0:c] = arr.toarray()[:,0:c]
        for i in range(0, c + dc, size):
            arr_part.append(
                sparse.csr_matrix(
                    exp_arr[:,i:i+size]))

    if random:
        shuffle(arr_part)

    return arr_part

class DataGen(object):

    def __init__(self, file_path, batch_size, channel_size, width = 128*3, seed=4):
        self.file_path = file_path
        self.batch_size = batch_size
        self.channel_size = channel_size
        self.width = width
        self.seed = seed
        self._data = get_piano_rolls(file_path, width)
        np.random.seed(seed=seed)

    def __len__(self):
        return len(self._data)

    def gen_batch(self):
        indx = np.random.randint(0, high = len(self._data), size = self.batch_size)
        data = [self._data[i] for i in indx]
        return data

if __name__ == '__main__':
    data = DataGen('data/*/*.npz', 68, 1)
    print("loaded {0} datasets".format(len(data)))
