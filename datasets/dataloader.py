from utils.utils import get_weights
import datasets


class DataLoader():
    def __init__(self, args, dataset, setname, checkpoint_ms=None, results_dir=None):
        self.dataset = dataset
        self.setname = setname
        self.checkpoint_ms = checkpoint_ms
        self.results_dir = results_dir

        self.features_dict = {}
        self.gt_dict = {}
        self.eval_gt_dict = {}

        self.CP_dict = None

        self.feature_path = f'data/features/{self.dataset}/{args.i3d_training}'

        self.get_data(args)


    def get_data(self, args):
        dataloader = getattr(datasets, f'{self.dataset}')
        feature_loader = dataloader(args, self.feature_path, self.setname, self.results_dir)

        if args.eval_use_CP:
            self.CP_dict = feature_loader.CP_dict

        if self.dataset == 'phoenix14' and args.use_test and args.extract_save_pseudo_labels and (args.pseudo_label_type == 'PL' or args.pseudo_label_type == 'CP'):
            self.feature_path_test = f'data/features/{self.dataset}/{args.i3d_training}'
            feature_loader_test = dataloader(args, self.feature_path_test, 'test', self.results_dir)
            feature_loader.features_dict.update(feature_loader_test.features_dict)
            feature_loader.eval_gt_dict.update(feature_loader_test.eval_gt_dict)
            feature_loader.gt_dict.update(feature_loader_test.gt_dict)
            feature_loader.vid_list.extend(feature_loader_test.vid_list)

        self.features_dict = feature_loader.features_dict
        self.gt_dict = feature_loader.gt_dict
        self.eval_gt_dict = feature_loader.eval_gt_dict
        self.vid_list = feature_loader.vid_list
        self.num_classes = feature_loader.num_classes

        self.weights = args.weights
        if args.weights == 'opt':
            self.weights = get_weights(self.gt_dict)
