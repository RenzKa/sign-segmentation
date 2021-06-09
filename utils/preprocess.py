import pickle
import numpy as np
from math import ceil
from scipy.stats import norm
from pathlib import Path
import json

from utils.utils import get_labels_start_end_time

def gauss_dist(len_vector, mean, std):
    vec = np.round(norm.pdf(list(range(len_vector)), loc=mean, scale=std), 2)
    return vec/max(vec)
    

def find_boundaries(glosses, num_boundary_frames, boundaries_labels=None, uniform=0):
    boundaries = []
    signs = []
    for i, sen in enumerate(glosses):
        
        if len(sen) == 0:
            boundaries.append([0]*len(sen))
            continue
        boundaries.append([0]*len(sen))
        old_gloss = sen[0]
        for ix, gloss in enumerate(sen):
            if gloss != old_gloss:
                boundaries[-1] = [1 if i >= ix-round(num_boundary_frames/2) and i < ix+max(0, num_boundary_frames//2) else old for i,old in enumerate(boundaries[-1])]
            old_gloss = gloss
        if sum(boundaries[i]) == 0 and boundaries_labels != None:
            boundaries[i] = boundaries_labels[i]
        if uniform:
            signs.append((sum(boundaries[i][8:-7])//num_boundary_frames) + 1)   # hard coded in_frames 16 need to change

    if uniform:
        return boundaries, signs
    return boundaries

def dilate_boundaries(gt):
    eval_boundaries = []
    for item in gt:
        gt_temp = [0,0]+item+[0,0]
        con = 0
        for ix in range(2, len(item)+2):
            if con:
                con = 0 
                continue
            if gt_temp[ix] == 1 and gt_temp[ix+1] == 0 and gt_temp[ix+2] == 0:
                gt_temp[ix+1] = 1
                con = 1
            if gt_temp[ix] == 1 and gt_temp[ix-1] == 0 and gt_temp[ix-2] == 0:
                gt_temp[ix-1] = 1
        eval_boundaries.append(gt_temp[2:-2])
    return eval_boundaries


def find_boundaries_eval(glosses, num_boundary_frames, boundaries_labels=None):
    # broader boundaries but restricted so that not two boundaries melted togehter
    boundaries = []

    for i, sen in enumerate(glosses):
        #count instance length
        
        if len(sen) == 0:
            boundaries.append([])
            continue
        old_gloss = sen[0]
        len_instances = []
        count_frames = 0
        for ix, gloss in enumerate(sen):
            
            if gloss == old_gloss:
                count_frames += 1
                old_gloss = gloss

            if gloss != old_gloss and ix == len(sen)-1:
                len_instances.append(count_frames)
                len_instances.append(1)

            elif gloss != old_gloss or ix == len(sen)-1:
                len_instances.append(count_frames)
                count_frames = 1
                old_gloss = gloss

        assert sum(len_instances)==len(sen)
        boundaries.append(np.zeros(len(sen), dtype=int))
        
        old_gloss = sen[0]
        counter = 0
        for ix, gloss in enumerate(sen):
            if gloss != old_gloss:
                counter += 1
                start = ix - min(num_boundary_frames//2, round(len_instances[counter-1]/4))
                end = ix + min(num_boundary_frames//2, round(len_instances[counter]/4))
                boundaries[-1][start:end] = 1

            old_gloss = gloss  
        
        if sum(boundaries[-1]) == 0:
            boundaries[-1] = boundaries_labels[i]

    return boundaries

def filter_silence(glosses, threshold):
    filtered_glosses = []
    for seq in glosses:
        if -1 in seq:
            subseq = np.zeros(len(seq), dtype=int)
            count_silence = 0

            for ix, item in enumerate(seq):
                if item != -1:
                    if count_silence > threshold:
                        start = ix-count_silence
                        subseq[start:ix] = -1
                        count_silence = 0
                    elif count_silence != 0:
                        start = ix-count_silence
                        if ix == count_silence:
                            subseq[start:start+count_silence//2] = item
                        else:
                            subseq[start:start+count_silence//2] = previous_item
                        subseq[start+count_silence//2:ix] = item
                        count_silence = 0

                    subseq[ix] = item
                    previous_item = item

                else:
                    count_silence += 1
            filtered_glosses.append(subseq)

        else:
            filtered_glosses.append(seq)

    return filtered_glosses
