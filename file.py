from math import log10, floor
import pretty_midi
from PIL import Image
import numpy as np
from scipy import sparse
import os

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

class MidiFile(object):
    def __init__(self, filepath, arrays):

        self._files = filepath
        self._arrays = arrays


    @classmethod
    def from_npz(cls, filepath):
        files = []
        arrays = []
        for f in filepath:
            fname, ext = os.path.splitext(f)
            files.append(fname)
            arr = sparse.load_npz(f)
            arrays.append(arr)

        return cls(files, arrays)
    
    @classmethod
    def from_mid(cls, filepath, verbose=True):
        tot_len = 0
        count_failed = 0
        files = []
        arrays = []

        if verbose:
            print("{:>5} {:>5}".format(
                'time', 'file'))

        for f in filepath:
            fname, ext = os.path.splitext(f)
            try:
                midi_data = pretty_midi.PrettyMIDI(f)
                if verbose:
                    print("{:>6} {:>15}".format(
                        round_sig(midi_data.get_end_time(), 4), f))
                tot_len += midi_data.get_end_time()
                files.append(fname)
                arr = midi_data.get_piano_roll()
                # normalize velocities
                arr = (255/np.max(arr)) * arr
                # store as sparse matrix to save space
                arrays.append(sparse.csr_matrix(arr))

            except:
                if verbose:
                    print('error with file {0}'.format(f))
                count_failed+=1

        if verbose:
            print("total length (sec): {0}".format(tot_len))
            print("failed: {0}".format(count_failed))

        return cls(files, arrays)


    def save_piano_rolls(self):
        for arr, file in zip(self._arrays, self._files):
            sparse.save_npz(file, arr)

    def view_array(self, ind, save_to = None):
            # TODO: change from RGB
            img = Image.fromarray(self._arrays[i], 'RGB')
            if save_to:
                img.save(save_to)
            else:
                img.show()











