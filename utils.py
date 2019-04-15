from file import MidiFile
import glob
import numpy as np
from scipy import sparse
from random import shuffle

def save_piano_rolls(file_path):

    f = glob.glob(file_path)
    mf = MidiFile.from_mid(f)
    mf.save_piano_rolls()

def get_piano_rolls(file_path, size = 128, random = True):

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

def gen_data(file_path):

if __name__ == '__main__':
    data = get_piano_rolls('data/*/*.npz')
    print("loaded {0} datasets".format(len(data)))
