import argparse
import pickle
from math import floor, ceil
import numpy as np
import cv2
import seaborn as sns
from pathlib import Path

import ruptures as rpt
from zsvision.zs_utils import loadmat


def get_CP_dict(feature_dict, vid_list):
    """
    Function for CP Baseline
    Calculate changepoints for whole epsiode.

    Args:
        features_dict: dictionary of i3d features to calculate changepoints
        vid_list: list of video names

    Returns:
        dictionary of changepoints (key: video names; value: changepoints)
    """
    CP_dict = {}
    model ='l2'
    pen = 80
    jump = 2

    for vid in vid_list:
        features = feature_dict[vid]
        if len(features) < 2:
            CP_dict[vid] = np.zeros(len(features))
            continue

        algo = rpt.Pelt(model=model, jump=jump).fit(features)
        res = algo.predict(pen=pen)
        res_np = [1 if ix in res else 0 for ix in range(len(features))]

        CP_dict[vid] = np.asarray(res_np)

    return CP_dict
