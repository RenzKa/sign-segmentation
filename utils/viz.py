import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn

from bokeh.palettes import d3, Turbo256
import seaborn as sns

from utils.utils import torch_to_list

all_colors = []
for ii in range(22):
    all_colors += (
        d3["Category20"][20] + d3["Category20b"][20] + d3["Category20c"][20]
    )

def viz_results_paper(gt, pred, name='test', pred_prob=None, pred_actsig=None, pred_prob_actsig=None):
    fr_width = 30
    height = 420
    width = len(gt)*fr_width+240
    image = np.ones((height,width,3), np.uint8)*255

    palette = [(0.9,0.9,0.9), sns.color_palette("colorblind", 10)[0]]   # deep, muted, pastel, bright, dark, colorblind
    palette2 = [(0.9,0.9,0.9), sns.color_palette("pastel", 10)[0]]   # deep, muted, pastel, bright, dark, colorblind

    shift = 200
    item_old = gt[0]
    count = 1
    color = (0,0,255)
    cv2.putText(image,'gt',(5,150), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,0),3)
    for ix, item in enumerate(gt):
        item_old = item
        color = tuple([palette[item][i] *255 for i in [2,1,0]])
        image[20:200, fr_width*(ix+1)+shift:fr_width*(ix+1)+shift-1+fr_width] = color

    item_old = pred[0]
    count = 1
    color = (0,0,255)
    cv2.putText(image,'ms',(5,350), cv2.FONT_HERSHEY_SIMPLEX, 5,(0,0,0),3)
    for ix, item in enumerate(pred):
        item_old = item
        color = tuple([palette2[item][i] *255 for i in [2,1,0]])
        image[220:400, fr_width*(ix+1)+shift:fr_width*(ix+1)+shift-1+fr_width] = color

    if pred_prob is not None:
        old_item = pred_prob[0]
        for ix, item in enumerate(pred_prob):
            y = 400-int(pred_prob[ix]*(400-220))
            x = fr_width*(ix+1)+shift+fr_width//2
            if ix > 0:
                image = cv2.line(image, (x_old, y_old), (x,y), (0,0,0), 4)
            x_old = x
            y_old = y

    cv2.imwrite(f'{name}_paper.jpg', image)