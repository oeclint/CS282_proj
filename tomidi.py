"""
Utility function for converting an audio file
to a pretty_midi.PrettyMIDI object. Note that this method is nowhere close
to the state-of-the-art in automatic music transcription.
This just serves as a fun example for rough
transcription which can be expanded on for anyone motivated.
"""
from __future__ import division
import sys
import argparse
import numpy as np
import pretty_midi
import os
from scipy.ndimage import imread


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

if __name__ == '__main__':
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Transcribe Audio file to MIDI file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img', action='store',
                         help='path to the input img')

    parser.add_argument('--pad', default=2, type=int, action='store',
                         help='padding in input img')

    args = parser.parse_args()

    img = imread(args.img)

    pre, ext = os.path.splitext(args.img)

    H1, W1, C1 = img.shape

    H = 384
    W = 48
    C = 3

    top = 40
    bot = 87

    non_pad_img = []

    j1 = args.pad
    j2 = H1 - args.pad

    i1 = args.pad
    i2 = i1 + W

    while i1 < W1: 
        img_slc = img[j1:j2, i1:i2, :]
        img_slc = np.moveaxis(img_slc, -1, 0).reshape(H*C,W)
        non_pad_img.append(img_slc.T)
        i1 = i2 + args.pad
        i2 = i1 + W

    for i, arr in enumerate(non_pad_img):
        h, w = arr.shape
        pn = np.zeros((128,w))
        pn[top:(bot+1)] = arr*(127/255)
        pm = piano_roll_to_pretty_midi(pn)
        f = '{0}-{1}.mid'.format(pre,i)
        pm.write(f)