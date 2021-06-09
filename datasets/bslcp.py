import random
from pathlib import Path
import pickle
import numpy as np

from zsvision.zs_utils import loadmat

from self_labelling.changepoint_detection import get_CP_dict
from utils.preprocess import gauss_dist, dilate_boundaries


class bslcp():
    def __init__(
            self,
            args,
            features_path,
            setname,
            results_dir = None,
            ):

        self.setname = setname
        self.features_path = f'{features_path}/{self.setname}'
        if setname == 'train':
            self.setvalue = 0
        elif setname == 'eval':
            self.setvalue = 1
        elif setname == 'test':
            self.setvalue = 2

        # load info file
        self.info_file = f'data/info/bslcp/info.pkl'
        with open(self.info_file, 'rb') as f:
            self.info_data = pickle.load(f)

        # create list of video names
        self.vid_list = [video_name for ix, video_name in enumerate(self.info_data['videos']['name']) if self.info_data['videos']['split'][ix] == self.setvalue]

        self.get_features(args)


    def get_features(self, args):
        self.num_classes = 2

        # classification
        gt_list = [glosses for ix, glosses in enumerate(self.info_data['videos']['alignments']['boundaries']) if self.info_data['videos']['split'][ix] == self.setvalue]
        gt_list_eval = dilate_boundaries(gt_list)

        # regression
        if args.regression:
            self.num_classes = 1
            if self.setname == 'train':
                std = args.std
                gt_gauss = []
                for i, gt in enumerate(gt_list):
                    label_old = gt[0]
                    boundary_frames = []
                    gauss = []
                    for ix, label in enumerate(gt):
                        if label == 1:
                            boundary_frames.append(ix)
                        else:
                            if label_old:
                                gauss.append(gauss_dist(len(gt), np.mean(np.asarray(boundary_frames)), std))
                            boundary_frames = []
                        label_old = label

                    gauss = np.asarray(gauss)
                    gt_gauss.append(list(np.amax(gauss, axis=0)))
                gt_list = gt_gauss.copy()
                gt_list_eval = dilate_boundaries([[int(round(item)) for item in sublist] for sublist in gt_list])


        # read i3d features
        features_data = loadmat(Path(f'{self.features_path}/features.mat'))
        features = features_data['preds']

        if args.feature_normalization:
            features = np.nan_to_num((features - np.mean(features, axis=0)) / np.std(features))

        # see if length of extracted features fit to the videos (input window size of i3d training (args.num_in_frames) leads to less features than frames)
        num_frames_list = [max(t-(args.num_in_frames-1), 0) for ix, t in enumerate(self.info_data['videos']['videos']['T']) if self.info_data['videos']['split'][ix] == self.setvalue]
        ges_frames = sum(num_frames_list)
        assert ges_frames == features.shape[0]
        assert args.features_dim == features.shape[1]

        # assign features to videos
        self.features_dict = {}
        for ix, video in enumerate(self.vid_list):
            start = sum(num_frames_list[:ix])
            end = start + num_frames_list[ix]
            self.features_dict[video] = features[start:end]

        # assign ground truth to videos
        self.gt_dict = {}
        for ix, video in enumerate(self.vid_list):
            self.gt_dict[video] = gt_list[ix][int(args.num_in_frames/2):-int(args.num_in_frames/2)+1]
            assert len(self.gt_dict[video]) == len(self.features_dict[video])

        self.eval_gt_dict = {}
        for ix, video in enumerate(self.vid_list):
            self.eval_gt_dict[video] = gt_list_eval[ix][int(args.num_in_frames/2):-int(args.num_in_frames/2)+1]
            assert len(self.eval_gt_dict[video]) == len(self.features_dict[video])

        # delete videos with less than 16 frames:
        del_list = np.where(np.asarray(num_frames_list) == 0)
        for item in del_list[0]:
            del self.gt_dict[self.vid_list[item]]
            del self.features_dict[self.vid_list[item]]
        self.vid_list = [vid for ix, vid in enumerate(self.vid_list) if ix not in list(del_list[0])]

        # choose random subset for prediction
        if args.action == 'predict' and args.test_subset != -1:
            self.vid_list = random.choices(self.vid_list, k=args.test_subset)

        if args.eval_use_CP:
            self.CP_dict = get_CP_dict(self.test_features_dict, self.test_vid_list, self.test_gt_dict)
