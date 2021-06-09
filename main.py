import argparse
import random
import torch
import numpy as np
from pathlib import Path

from self_labelling.PL_CP_fusion_methods import get_save_local_fusion, merge_PL_CP, CMPL, extract_CP
from utils.utils import create_folders
from model import Trainer
from datasets.dataloader import DataLoader
from batch_gen import BatchGenerator


def main(args, device, model_load_dir, model_save_dir, results_save_dir):

    if args.action == 'train' and args.extract_save_pseudo_labels == 0:
        # load train dataset and test dataset
        print(f'Load train data: {args.train_data}')
        train_loader = DataLoader(args, args.train_data, 'train')
        print(f'Load test data: {args.test_data}')
        test_loader = DataLoader(args, args.test_data, 'test')

        print(f'Start training.')
        trainer = Trainer(
                    args.num_stages,
                    args.num_layers,
                    args.num_f_maps,
                    args.features_dim,
                    train_loader.num_classes,
                    device,
                    train_loader.weights,
                    model_save_dir
                    )

        eval_args = [
            args,
            model_save_dir,
            results_save_dir,
            test_loader.features_dict,
            test_loader.gt_dict,
            test_loader.eval_gt_dict,
            test_loader.vid_list,
            args.num_epochs,
            device,
            'eval',
            args.classification_threshold,
        ]

        batch_gen = BatchGenerator(
            train_loader.num_classes,
            train_loader.gt_dict,
            train_loader.features_dict,
            train_loader.eval_gt_dict
            )

        batch_gen.read_data(train_loader.vid_list)
        trainer.train(
            model_save_dir,
            batch_gen,
            args.num_epochs,
            args.bz,
            args.lr,
            device,
            eval_args,
            pretrained=model_load_dir)

    elif args.extract_save_pseudo_labels and args.pseudo_label_type != 'PL':
        # extract/ generate pseudo labels and save in "data/pseudo_labels"
        print(f'Load test data: {args.test_data}')
        test_loader = DataLoader(args, args.test_data, args.extract_set, results_dir=results_save_dir)
        print(f'Extract {args.pseudo_label_type}')
        
        if args.pseudo_label_type == 'local':
            get_save_local_fusion(args, test_loader.features_dict, test_loader.gt_dict)
        elif args.pseudo_label_type == 'merge':
            merge_PL_CP(args, test_loader.features_dict, test_loader.gt_dict)
        elif args.pseudo_label_type == 'CMPL':
            CMPL(args, test_loader.features_dict, test_loader.gt_dict)
        elif args.pseudo_label_type == 'CP':
            extract_CP(args, test_loader.features_dict)
        
        print('Self labelling process finished')


    else:
        print(f'Load test data: {args.test_data}')
        test_loader = DataLoader(args, args.test_data, args.extract_set, results_dir=results_save_dir)

        if args.extract_save_pseudo_labels and args.pseudo_label_type == 'PL':
            print(f'Extract {args.pseudo_label_type}')
            extract_save_PL = 1
        else:
            print(f'Start inference.')
            extract_save_PL = 0

        trainer = Trainer(
            args.num_stages,
            args.num_layers,
            args.num_f_maps,
            args.features_dim,
            test_loader.num_classes,
            device,
            test_loader.weights,
            results_save_dir)

        trainer.predict(
            args,
            model_load_dir,
            results_save_dir,
            test_loader.features_dict,
            test_loader.gt_dict,
            test_loader.eval_gt_dict,
            test_loader.vid_list,
            args.num_epochs,
            device,
            'test',
            args.classification_threshold,
            uniform=args.uniform,
            save_pslabels=extract_save_PL,
            CP_dict=test_loader.CP_dict,
            )


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--action', default='train', choices=['train', 'predict'])
    parser.add_argument('--extract_set', default='train', choices=['train', 'test'])
    parser.add_argument(
        '--train_data',
        default='bslcp',
        choices=['bslcp', 'phoenix14']
        )

    parser.add_argument(
        '--test_data',
        default='bslcp',
        choices=['bslcp', 'phoenix14']
        )

    parser.add_argument(
        '--i3d_training',
        default='i3d_kinetics_bslcp_981'
        )

    parser.add_argument('--num_in_frames', default=16, type=int)
    parser.add_argument('--folder', default='', type=str, help="folder to save the results")

    ### viz
    parser.add_argument('--test_subset', default=-1, type=int, help='use only a subset of the test set for evaluation and visualization')
    parser.add_argument('--viz_results', action='store_true', help="save visualizations of results and gt for each video sequence")

    ### target_finetuning settings
    parser.add_argument('--extract_save_pseudo_labels', default=0, type=int, choices=[0,1], help="extract and save pseudo_labels of type pseudo_label_type")
    parser.add_argument('--pseudo_label_type', default='CP', choices=['CMPL', 'merge', 'local', 'PL', 'CP'], type=str)

    parser.add_argument('--use_test', action='store_true', help="append test set to extract_set (for combined train and test set")

    parser.add_argument('--pretrained', default='', type=str, help="Path to pretrained model")

    parser.add_argument('--use_pseudo_labels', action='store_true', help="Use already extracted pseudo-labels for training ")
    parser.add_argument('--load_label_path', default='', type=str, help="Path to pseudo-labels")
    # parser.add_argument('--PL_info_savefolder', default=False)

    parser.add_argument('--eval_use_CP', action='store_true', help="CP Baseline, use changepoints as predictions (without training)")  

    ### fusion
    parser.add_argument('--local_fusion_model', default='l2')
    parser.add_argument('--local_fusion_pen', default=80, type=int)
    parser.add_argument('--local_fusion_jump', default=2, type=int)
    parser.add_argument('--local_fusion_th_min', default=13, type=int)
    parser.add_argument('--local_fusion_th_max', default=60, type=int)

    parser.add_argument('--merge_model', default='l2')
    parser.add_argument('--merge_pen', default=80, type=int)
    parser.add_argument('--merge_jump', default=2, type=int)

    parser.add_argument('--CMSL_model', default='l2')
    parser.add_argument('--CMSL_pen', default=[100,100])
    parser.add_argument('--CMSL_jump', default=2, type=int)
    parser.add_argument('--CMSL_th_insert', default=4, type=int)
    parser.add_argument('--CMSL_th_refine', default=4, type=int)

    ### MS-TCN HYPERPARAMETER
    parser.add_argument('--num_stages', default=4, type=int)
    parser.add_argument('--num_layers', default=10, type=int)
    parser.add_argument('--num_f_maps', default=64, type=int)
    parser.add_argument('--features_dim', default=1024, type=int)
    parser.add_argument('--bz', default=8, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--extract_epoch', default=10, type=int)
    parser.add_argument('--weights', default='opt', help="None, [1., 5.], 'opt'")

    parser.add_argument('--uniform', default=0, type=int)
    parser.add_argument('--regression', default=0, type=int)
    parser.add_argument('--std', default=1, type=int)
    parser.add_argument('--classification_threshold', default=0.5, type=float)

    #### Other settings
    parser.add_argument('--feature_normalization', default=0, type=int)
    parser.add_argument('--num_boundary_frames', default=2, type=int)

    args = parser.parse_args()
    # if args.PL_info_savefolder is False:
    args.PL_info_savefolder = str(Path(args.load_label_path).stem)
    if args.pretrained == '':
        args.pretrained = False

    # set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create models and save args
    model_load_dir, model_save_dir, results_save_dir = create_folders(args)
    main(args, device, model_load_dir, model_save_dir, results_save_dir)
