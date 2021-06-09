import random
from pathlib import Path
import pickle
import numpy as np

from zsvision.zs_utils import loadmat

from self_labelling.changepoint_detection import get_CP_dict
from utils.preprocess import gauss_dist, dilate_boundaries, filter_silence, find_boundaries, find_boundaries_eval


class phoenix14():
    def __init__(
        self, 
        args,
        features_path,
        setname,
        results_dir,
        ):

        self.setname = setname
        self.features_path = f'{features_path}/{self.setname}'
        self.test_features_path = f'{features_path}/test'
        
        if setname == 'train':
            self.setvalue = 0
        elif setname == 'eval':
            self.setvalue = 1
        elif setname == 'test':
            self.setvalue = 2

        # load info file
        self.info_file = 'data/info/phoenix14/info.pkl'
        with open(self.info_file, 'rb') as f:
            self.info_data = pickle.load(f)

        # create list of video names
        self.vid_list = [video_name for ix, video_name in enumerate(self.info_data['videos']['name']) if self.info_data['videos']['split'][ix] == self.setvalue]
        # for training with self labels
        self.test_vid_list = [video_name for ix, video_name in enumerate(self.info_data['videos']['name']) if self.info_data['videos']['split'][ix] == 2]

        self.get_features(args)


    def get_features(self, args):
        self.num_classes = 2

        #extract boundary frames from frame-level annotations
        glosses = filter_silence(self.info_data['videos']['alignments']['gloss_id'], 8)
        boundaries = find_boundaries(glosses, args.num_boundary_frames)
        boundaries_eval = find_boundaries_eval(glosses, 4) # boundary width 4 for eval

        # assign gt to test and train set
        gt_list = [glosses for ix, glosses in enumerate(boundaries) if self.info_data['videos']['split'][ix] == self.setvalue]
        gt_list_eval = [glosses for ix, glosses in enumerate(boundaries_eval) if self.info_data['videos']['split'][ix] == self.setvalue]

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
                    
                    if len(gauss) == 0:
                        print('d')
                    gauss = np.asarray(gauss)
                    gt_gauss.append(list(np.amax(gauss, axis=0)))    
                gt_list = gt_gauss.copy()            
                gt_list_eval = dilate_boundaries([[int(round(item)) for item in sublist] for sublist in gt_list])

        # read i3d features
        features_data = loadmat(Path(f'{self.features_path}/features.mat'))
        features = features_data['preds']

        if args.feature_normalization:
            features = np.nan_to_num((train_features - np.mean(train_features, axis=0)) / np.std(train_features))

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
            if end-start < 2:
                continue
            self.features_dict[video] = features[start:end]

        del_videos = []
        # assign ground truth to videos
        self.gt_dict = {}
        for ix, video in enumerate(self.vid_list):
            if len(gt_list[ix]) == 0:
                del_videos.append(video)
                continue
            self.gt_dict[video] = gt_list[ix][int(args.num_in_frames/2):-int(args.num_in_frames/2)+1]
            assert len(self.gt_dict[video]) == len(self.features_dict[video])

        self.eval_gt_dict = {}
        for ix, video in enumerate(self.vid_list):
            if len(gt_list[ix]) == 0:
                del_videos.append(video)
                continue
            self.eval_gt_dict[video] = gt_list_eval[ix][int(args.num_in_frames/2):-int(args.num_in_frames/2)+1]
            assert len(self.eval_gt_dict[video]) == len(self.features_dict[video])

        # delete videos with less than 16 frames:
        self.vid_list = [vid for ix, vid in enumerate(self.vid_list) if vid not in del_videos]

        # choose random subset for prediction
        if args.action == 'predict' and args.test_subset != -1:
            self.vid_list = random.sample(self.vid_list, k=args.test_subset)

        if args.eval_use_CP:
            self.CP_dict = get_CP_dict(self.features_dict, self.vid_list)

        # if args.pseudo_label_type == 'CP':
        #     self.gt_dict = get_CP_dict(self.features_dict, self.vid_list)
        #     self.gt_dict_eval = {key: dilate_boundaries([item])[0] for key, item in self.gt_dict.items()}

        # if args.extract_save_pseudo_labels and args.pseudo_label_type == 'PL' and args.use_test:
        #     self.vid_list.extend(self.test_vid_list)
        #     test_features_data = loadmat(Path(f'{self.test_features_path}/features.mat'))
        #     test_features = features_data['preds']
        #     test_num_frames_list = [max(t-(args.num_in_frames-1), 0) for ix, t in enumerate(self.info_data['videos']['videos']['T']) if self.info_data['videos']['split'][ix] == 2]

        #     # assign features to videos
        #     self.test_features_dict = {}
        #     for ix, video in enumerate(self.test_vid_list):
        #         start = sum(test_num_frames_list[:ix])
        #         end = start + test_num_frames_list[ix]
        #         if end-start < 2:
        #             continue
        #         self.test_features_dict[video] = test_features[start:end]
        #     self.features_dict.update(self.test_features_dict)
            




        if (args.use_pseudo_labels or (args.extract_save_pseudo_labels and args.pseudo_label_type != 'PL' and args.pseudo_label_type != 'CP')) and self.setname == 'train':
            self.gt_dict = {}
            for vid in self.vid_list:
                # TODO
                episode = f"{vid.split('.')[0]}"
                #episode = f"{vid}"

                PL_root = args.load_label_path
                if args.regression:
                    self.num_classes = 1
                    load_label_path = f'{PL_root}/{episode}/scores.pkl'
                elif args.pseudo_label_type == 'CP':
                    load_label_path = f'{PL_root}/{episode}/pen_80.pkl'
                else:
                    load_label_path = f'{PL_root}/{episode}/preds.pkl'

                with open(load_label_path, 'rb') as f:
                    PL_labels = pickle.load(f)
                    assert len(PL_labels)==len(self.features_dict[vid])
                if args.pseudo_label_type == 'CP':
                    PL_labels = dilate_boundaries([PL_labels.tolist()])[0]
                self.gt_dict[vid] = np.asarray(PL_labels)
            self.eval_gt_dict = self.gt_dict.copy()

            if args.use_test:
                # load features
                test_features_data = loadmat(Path(f'{self.test_features_path}/features.mat'))
                test_features = features_data['preds']
                test_num_frames_list = [max(t-(args.num_in_frames-1), 0) for ix, t in enumerate(self.info_data['videos']['videos']['T']) if self.info_data['videos']['split'][ix] == 2]

                # assign features to videos
                self.test_features_dict = {}
                for ix, video in enumerate(self.test_vid_list):
                    start = sum(test_num_frames_list[:ix])
                    end = start + test_num_frames_list[ix]
                    if end-start < 2:
                        continue
                    self.test_features_dict[video] = test_features[start:end]

                self.gt_dict2 = {}
                for vid in self.test_vid_list:
                    episode = f"{vid.split('.')[0]}"

                    PL_root = args.load_label_path
                    if args.regression:
                        self.num_classes = 1
                        load_label_path = f'{PL_root}/{episode}/scores.pkl'
                    elif args.pseudo_label_type == 'CP':
                        load_label_path = f'{PL_root}/{episode}/pen_80.pkl'
                    else:
                        load_label_path = f'{PL_root}/{episode}/preds.pkl'

                    with open(load_label_path, 'rb') as f:
                        PL_labels = pickle.load(f)
                        assert len(PL_labels)==len(self.test_features_dict[vid])
                    if args.pseudo_label_type == 'CP':
                        PL_labels = dilate_boundaries([PL_labels.tolist()])[0]
                    self.gt_dict2[vid] = np.asarray(PL_labels)

                self.eval_gt_dict2 = self.gt_dict2.copy()

                self.gt_dict.update(self.gt_dict2)
                self.eval_gt_dict.update(self.eval_gt_dict2)
                self.vid_list.extend(self.test_vid_list)
                self.features_dict.update(self.test_features_dict)

        # elif args.use_cp_pslabels:
        #     self.train_gt_dict = pickle.load(open(f'/users/katrin/coding/libs/segmentation/ms-tcn_BSLCP/ms-tcn_SL/exps/exp_cvpr_final1/PseudoLabels/changepoint/phoenix/cp_dict.pkl', "rb"))
        #     self.eval_train_gt_dict = self.train_gt_dict.copy()
        #     self.train_features_dict.update(self.test_features_dict)
        #     if args.use_test:
        #         self.train_vid_list.extend(self.test_vid_list)
                

                
