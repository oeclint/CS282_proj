from file import MidiFile
import glob
import numpy as np
from scipy import sparse
from random import shuffle
from collections import deque
from PIL import Image
import scipy.misc

from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
    def __len__(self):
        return len(self.data)

def save_piano_rolls(file_path):

    f = glob.glob(file_path)
    mf = MidiFile.from_mid(f)
    mf.save_piano_rolls()

def get_piano_rolls(file_path, size_per_channel = 128*3, num_channels = 1, octave4 = True):

    f = glob.glob(file_path)
    mf = MidiFile.from_npz(f)

    if octave4:
        min_top = 127
        max_bot = 0
    else:
        min_top = 0
        max_bot = 127

    data_arr = deque()

    for mf_arr in mf._arrays:
        r, c = mf_arr.shape
        size = size_per_channel * num_channels
        dc = (size - c % size) % size
        arr = np.zeros((r, c + dc))
        arr[:,0:c] = mf_arr.toarray()[:,0:c]
        for i in range(0, c + dc, size):
            #channel_arr = np.zeros((num_channels, r, size_per_channel))
            channel_arr = []
            for cx, j in enumerate(range(i, i + size, size_per_channel)):
                if octave4:
                    octshft = OctaveShifter(arr[:,j : j + size_per_channel])
                    #channel_arr[cx] = octshft.shift()
                    shifted = octshft.shift_to_4_octaves()
                    channel_arr.append(sparse.csr_matrix(shifted))
                    if np.any(shifted):
                        min_top = min(min_top,octshft.top)
                        max_bot = max(max_bot,octshft.bot)
                else:
                    channel_arr.append(sparse.csr_matrix(
                        arr[:,j : j + size_per_channel]))
                    
            data_arr.append(channel_arr)

    return data_arr, min_top, max_bot

class DataGen(object):

    def __init__(self, file_path, batch_size, num_channels, width = 128*3, seed=4):
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.width = width
        self.seed = seed
        np.random.seed(seed=seed)
        self._data, self.top, self.bot = get_piano_rolls(file_path, width, num_channels)

    def clip(self):
        data = []

        while(self._data):
            piano_roll = self._data.popleft()
            channels = np.stack([pr.toarray() for pr in piano_roll])
            channels = channels[:,self.top:(self.bot+1),:]
            data.append([sparse.csr_matrix(channels[i]) 
                for i in range (len(channels))])

        self._data = data

    def write(self, path):
        for i, piano_roll in enumerate(self._data):
            #from PIL import Image
            #im = Image.fromarray(np.moveaxis(piano_roll, 0, -1))
            #im.save("your_file.jpeg")
            channels = np.stack([pr.toarray() for pr in piano_roll])
            arr = np.moveaxis(channels, 0, -1)
            arr = np.moveaxis(arr, 0, 1)
            scipy.misc.imsave('{0}-{1}.png'.format(path,i), arr)

    def __len__(self):
        return len(self._data)

    def gen_batch(self):
        indx = np.random.randint(0, high = len(self._data), size = self.batch_size)
        data = [np.stack([d.toarray() for d in self._data[i]]) for i in indx]
        return np.stack(data)

class OctaveShifter(object):

    def __init__(self, arr):
        rows, cols = arr.shape
        self.arr = arr
        self.top = 0
        self.bot = rows - 1
        for i in range(rows):
            if np.any(arr[i]):
                self.top = i
                break
        for i in reversed(range(rows)):
            if np.any(arr[i]):
                self.bot = i
                break
        self.center = rows // 2
           
    @property
    def mean(self):
        return (self.bot + self.top + 1)//2
   
    @property
    def next_mean_u(self):
           
        next_bot = self.bot - 12
        next_top = self.top - 12
              
        return (next_bot + next_top + 1)//2
   
    @property
    def next_mean_d(self):
       
       next_bot = self.bot + 12
       next_top = self.top + 12

       return (next_bot + next_top + 1)//2

    def shift_center(self):
       
        mean = self.mean
       
        if mean > self.center:
            # roll up
            mean_u = self.mean
            next_mean_u = self.next_mean_u

            while(abs(next_mean_u-self.center) < abs(mean_u-self.center)):
                self.arr = np.roll(self.arr, -12, axis=0)
                self.top -= 12
                self.bot -= 12
                mean_u = self.mean
                next_mean_u = self.next_mean_u
               
        if mean < self.center:
            # roll down
            mean_d = self.mean
            next_mean_d = self.next_mean_d

            while(abs(next_mean_d-self.center) < abs(mean_d-self.center)):
                self.arr = np.roll(self.arr, 12, axis=0)
                self.top += 12
                self.bot += 12
                mean_d = self.mean
                next_mean_d = self.next_mean_d

        return self.arr

    def shift_to_4_octaves(self):

        self.top = 40
        self.bot = 87

        r, c = self.arr.shape

        for j in range(c):
            for i in range(r):
                val = self.arr[i][j]
                if val:
                    if i < self.top:
                        # go down
                        curr_pos = i
                        while(curr_pos<self.top):
                            curr_pos += 12

                        self.arr[curr_pos][j] = val

                    if i > self.bot:
                        # go up
                        curr_pos = i
                        while(curr_pos>self.bot):
                            curr_pos -= 12

                        self.arr[curr_pos][j] = val

        return self.arr

if __name__ == '__main__':
#    save_piano_rolls('data/*/*.mid')    
    import os
    from sklearn.model_selection import train_test_split
    data = DataGen('data/*/*.npz', 68, 3)
    data.clip()
    root = 'PixelCNN/datasets/pianorolls'
    files = glob.glob(os.path.join(root,'**','*.png'), recursive = True)
    for f in files:
        os.remove(f)
    p = os.path.join(root,'d')
    trainp = os.path.join(root, 'train', 'data')
    testp = os.path.join(root, 'test', 'data')
    data.write(p)
    print('top:', data.top, 'bot:', data.bot)
    print("loaded {0} datasets".format(len(data)))
    print(data.gen_batch().shape)
    trains, tests = train_test_split(glob.glob(p+'*.png'), test_size = 0.20)
    for train in trains:
        fname = train.split(os.sep)[-1]
        os.rename(train, os.path.join(trainp, fname))

    for test in tests:
        fname = test.split(os.sep)[-1]
        os.rename(test, os.path.join(testp, fname))



