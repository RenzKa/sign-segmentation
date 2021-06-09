import os
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm

import ruptures as rpt
import numpy as np
import cv2
import seaborn as sns

from zsvision.zs_utils import memcache
from utils.utils import get_labels_start_end_time
from utils.preprocess import dilate_boundaries

def extract_CP(args, features_dict):
    """
    Calculate and save Changepoints.
    Saves one .pkl label file per episode/ video.

    Args:
        args: parser arguments
        features_dict: dictionary of i3d features to calculate changepoints
    """
    pen = 80
    for vid in tqdm(features_dict.keys()):
        features = features_dict[vid]
        changepoints = []

        # check if CP for this episode with given setting (pen,..) already exists
        if not os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{pen}.pkl"):         
            algo = rpt.Pelt(model=args.merge_model, jump=args.merge_jump).fit(features)
            res = algo.predict(pen=pen)
            CP = [1 if ix in res else 0 for ix in range(len(features))]
            if not os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}"):
                os.makedirs(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}")
            pickle.dump(np.asarray(CP), open(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{pen}.pkl", "wb"))


def get_save_local_fusion(args, features_dict, PL_dict):
    """
    Calculate local fusion and save new pseudo-labels. 
    Changepoint detection only in long segments without boundaries.
    Saves one .pkl label file per episode/ video.

    Args:
        args: parser arguments
        features_dict: dictionary of i3d features to calculate changepoints
        PL_dict: pre-extracted pseudo-labels as startpoint
    """

    for vid in features_dict.keys():
        features = features_dict[vid]
        PL = PL_dict[vid]
        localfusion_labels = np.asarray(PL.copy())

        labels, starts, ends = get_labels_start_end_time(PL,[1])
        for ix in range(len(labels)):
            length = ends[ix] - starts[ix]
            start = starts[ix]
            end = ends[ix]
            if length > args.local_fusion_th_min and length < args.local_fusion_th_max:
                algo = rpt.Pelt(model=args.local_fusion_model, jump=args.local_fusion_jump).fit(features[start:end])
                res = algo.predict(pen=args.local_fusion_pen)
                res_np = [1 if ix in res else 0 for ix in range(end-start)]

                localfusion_labels[start:end] = res_np

        save_path = f"data/pseudo_labels/local_fusion/{args.test_data}/{args.i3d_training}/local_{args.local_fusion_th_min}_{args.local_fusion_th_max}_{args.local_fusion_pen}/seed_{args.seed}/{vid.split('.')[0]}"
         
        if not os.path.exists(f"{save_path}"):
            Path(f"{save_path}").mkdir(parents=True, exist_ok=True)
        pickle.dump(localfusion_labels, open(f"{save_path}/preds.pkl", "wb"))



def merge_PL_CP(args, features_dict, PL_dict):
    """
    Calculate "merge" fusion strategy and save new pseudo-labels. 
    Changepoint detection for whole epsiode and take union of PL and CP.
    Saves one .pkl label file per episode/ video.

    Args:
        args: parser arguments
        features_dict: dictionary of i3d features to calculate changepoints
        PL_dict: pre-extracted pseudo-labels as startpoint
    """
    for vid in features_dict.keys():
        features = features_dict[vid]
        PL = PL_dict[vid]
        changepoints = []

        # check if CP for this episode with given setting (pen,..) already exists
        if os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{args.merge_pen}.pkl"):
            cp = pickle.load(open(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{args.merge_pen}.pkl", "rb"))
            changepoints.append(cp)
        else:            
            algo = rpt.Pelt(model=args.merge_model, jump=args.merge_jump).fit(features)
            res = algo.predict(pen=args.merge_pen)
            CP = [1 if ix in res else 0 for ix in range(len(features))]
            changepoints.append(np.asarray(CP))
            if not os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}"):
                os.makedirs(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}")
            pickle.dump(np.asarray(CP), open(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{args.merge_pen}.pkl", "wb"))
        CP = np.asarray(dilate_boundaries([list(changepoints[0])])[0])

        merges = CP|PL

        save_path = f"data/pseudo_labels/PL_CP_merge/{args.test_data}/{args.i3d_training}/merge_{args.merge_pen}/seed_{args.seed}/{vid.split('.')[0]}"
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(merges, open(f"{save_path}/preds.pkl", "wb"))


def CMPL(args, features_dict, PL_dict):
    """
    Calculate "CMPL" fusion strategy and save new pseudo-labels. 
    Changepoint detection for whole epsiode and calculate insertion and refinement step.
    Saves one .pkl label file per episode/ video.

    Args:
        args: parser arguments
        features_dict: dictionary of i3d features to calculate changepoints
        PL_dict: pre-extracted pseudo-labels as startpoint
    """
    for vid in tqdm(features_dict.keys()):
        features = features_dict[vid]
        PL = PL_dict[vid]
        CP = []
        for pen in args.CMSL_pen:
            # check if CP for this episode with given setting (pen,..) already exists
            if os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{pen}.pkl"):
                cp_np = pickle.load(open(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{pen}.pkl", "rb"))
                CP.append(cp_np)
            else:
                # TODO: PCA to speed up?
                algo = rpt.Pelt(model=args.CMSL_model, jump=args.CMSL_jump).fit(features)
                res = algo.predict(pen=pen)
                cp_list = [1 if ix in res else 0 for ix in range(len(features))]
                CP.append(np.asarray(cp_list))
                if not os.path.exists(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}"): #HACK
                    os.makedirs(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}")
                pickle.dump(np.asarray(cp_list), open(f"data/pseudo_labels/CP/{args.test_data}/{vid.split('.')[0]}/pen_{pen}.pkl", "wb"))

        # insertion
        SL_insertion = np.asarray(PL.copy())

        _, PL_starts, PL_ends = get_labels_start_end_time(PL, [0])
        _, CP_starts, _ = get_labels_start_end_time(CP[0], [0])
        
        if args.CMSL_th_insert < 9000:
            for cp_pos in CP_starts:
                insert = True
                for PL_start, PL_end in zip(PL_starts, PL_ends):
                    if abs(PL_start-cp_pos) < args.CMSL_th_insert:
                        insert = False
                    if abs(cp_pos-PL_end) < args.CMSL_th_insert:
                        insert = False
                if insert == True:
                    SL_insertion[cp_pos-1:cp_pos+2] = 1


        # refinement
        SL_refinement = SL_insertion.copy()

        if args.CMSL_th_refine < 9000:
            _, PL_starts, PL_ends = get_labels_start_end_time(SL_refinement, [0])
            _, CP_starts, CP_ends = get_labels_start_end_time(CP[1], [0])
            for cp_pos in CP_starts:
                refine = False
                PL_ix = -1
                dist = 1000
                for ix, (PL_start, PL_end) in enumerate(zip(PL_starts, PL_ends)):
                    if abs(PL_start-cp_pos) < args.CMSL_th_refine and abs(PL_start-cp_pos)<dist:
                        refine = True
                        dist = abs(PL_start-cp_pos)
                        PL_ix = ix
                    if abs(cp_pos-PL_end) < args.CMSL_th_refine and abs(cp_pos-PL_end)<dist:
                        refine = True
                        dist = abs(cp_pos-PL_end)
                        PL_ix = ix

                if refine == True:
                    mean_ps = (PL_ends[PL_ix] + PL_starts[PL_ix]) // 2
                    new_pos = (mean_ps + cp_pos) // 2
                    SL_refinement[PL_starts[PL_ix]:PL_ends[PL_ix]] = 0
                    SL_refinement[new_pos-1:new_pos+2] = 1

        save_path = f"data/pseudo_labels/CMPL/{args.test_data}/{args.i3d_training}/CMSL_{'_'.join([str(item) for item in args.CMSL_pen])}_{args.CMSL_jump}_{args.CMSL_th_insert}_{args.CMSL_th_refine}/seed_{args.seed}/{vid.split('.')[0]}"
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(SL_refinement, open(f"{save_path}/preds.pkl", "wb"))
