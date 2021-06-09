import pympi
import sys
import os
import datetime
import pickle

def get_labels_start_end_time(frame_wise_labels, bg_class=["Sign"]):
    """get list of start and end times of each interval/ segment.

    Args:
        frame_wise_labels: list of framewise labels/ predictions.
        bg_class: list of all classes in frame_wise_labels which should be ignored

    Returns:
        labels: list of labels of the segments
        starts: list of start times of the segments
        ends: list of end times of the segments
    """
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def torch_to_list(torch_tensor):
    return torch_tensor.cpu().numpy().tolist()


def get_num_signs(gt):
    """count number of signs.

    Args:
        gt: list of framewise labels/ predictions.

    Returns:
        number of signs
    """
    item_old = gt[0]
    count = 0
    for ix, item in enumerate(gt):
        if item_old == 0 and item == 1:
            count += 1
        if ix == len(gt)-1 and item != 1:
            count += 1

        item_old = item
    return count


def parse_glosses(anno_dir, target_tiers,):
    """Parse ELAN EAF files into a python dictionary.

    Args:
        anno_dir: the root directory containing all eaf files to be parsed.
        target_tiers: the names of the tiers (e.g. "RH-IDgloss") to be parsed.

    Returns:
        the parsed data
    """
    anno_paths = list(anno_dir.glob("*.eaf"))
    print(f"Found {len(anno_paths)} in {anno_dir}")
    times = []
    count = 0
    end_old = 0
    fps = 25
    for anno_path in anno_paths:
        eafob = pympi.Elan.Eaf(str(anno_path))
        for tier in target_tiers:
            for annotation in eafob.get_annotation_data_for_tier(tier):
                start, end = [(x) / 1000 for x in annotation[:2]]
                if end_old == 0:
                    end_old = round(end*fps)+1
                if round(start*fps) - end_old < 6:
                    while end_old < round(start*fps):
                        times.append(end_old)
                        end_old += 1
                times.append(round(start*fps))
                times.append(round(end*fps))
                end_old = round(end*fps)+1

    return times


def get_weights(gt_dict):
    """Calculate weights for weighted cross entropy loss.

    Args:
        gt_dict: dictionary of the gt labels.

    Returns:
        list of weigths per class
    """
    count_list = [0, 0]
    for _, item in gt_dict.items():
        item = list(item)
        count_list[1] += item.count(1)
        count_list[0] += item.count(0)

    weights = [1/ count_list[i] if count_list[i]!=0 else 1 for i in range(2)]
    weights_norm = [i/ sum(weights) for i in weights]
    return weights_norm


def save_args(args, save_folder, opt_prefix="opt"):
    """Save arguments as .txt and .pkl file.

    Args:
        args: parser arguments
        save_folder: path to folder
        opt_prefix: name of file
    """
    opts = vars(args)
    os.makedirs(save_folder, exist_ok=True)

    # Save to text
    opt_filename = f"{opt_prefix}.txt"
    opt_path = os.path.join(save_folder, opt_filename)
    with open(opt_path, "a") as opt_file:
        opt_file.write("====== Options ======\n")
        for k, v in sorted(opts.items()):
            opt_file.write(f"{str(k)}: {str(v)}\n")
        opt_file.write("=====================\n")
        opt_file.write(f"launched at {str(datetime.datetime.now())}\n")

    # Save as pickle
    opt_picklename = f"{opt_prefix}.pkl"
    opt_picklepath = os.path.join(save_folder, opt_picklename)
    with open(opt_picklepath, "wb") as opt_file:
        pickle.dump(opts, opt_file)
    print(f"Saved options to {opt_path}")


def create_folders(args):
    """Creates folder structure to save models/ results.

    Args:
        args: parser arguments.

    Returns:
        model_load_dir: Path to pretrained model (if specified, otherwise empty) 
        model_save_dir: Path to the folder where the model is saved
        results_save_dir: Path to the folder where the (inference) results are saved
    """
    num_stages = args.num_stages
    num_layers = args.num_layers
    num_f_maps = args.num_f_maps
    features_dim = args.features_dim
    bz = args.bz
    lr = args.lr
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data

    # create folder for save model and results
    std_str = ''
    if args.regression:
        train_type=f'regression/std_{args.std}'
    else:
        train_type='classification'

    if args.weights == None:
        weighted_str = 'unweighted'
    elif args.weights == 'opt':
        weighted_str = 'weighted_opt'
    else:
        weighted_str = f'weighted_{args.weights[0]}_{args.weights[1]}'

    if args.feature_normalization == 1:
        norm_str = '_normalized'
    else:
        norm_str = ''

    if args.use_pseudo_labels:
        if train_data == 'phoenix14':
            ssl_str = f"pseudo_labels/{args.pseudo_label_type}/train1_test{args.use_test}"
        elif train_data == 'bsl1k':
            ssl_str = f"pseudo_labels/{args.pseudo_label_type}/n_episodes{args.bsl1k_train_subset}_test{args.use_test}"
    else:
        ssl_str = 'supervised'

    # load pretrained model from given path
    if args.pretrained:
        model_load_dir = args.pretrained
    else:
        if args.action == 'predict':
            model_load_dir = f"./exps/{args.folder}/models/{train_type}/traindata_{train_data}/{args.i3d_training}/{ssl_str}/{num_stages}_{num_layers}_{num_f_maps}_{features_dim}_{bz}_{lr}_{weighted_str}/seed_{args.seed}/epoch-{str(args.extract_epoch)}.model"  #+args.split
        else:
            model_load_dir = ''
    if not os.path.exists(model_load_dir) and ((args.pretrained and args.action == 'train') or args.action == 'predict'):
        print(f'Pre-trained model not existing at: {model_load_dir}')
        sys.exit()

    model_save_dir = f"./exps/{args.folder}/models/{train_type}/traindata_{train_data}/{args.i3d_training}/{ssl_str}/{num_stages}_{num_layers}_{num_f_maps}_{features_dim}_{bz}_{lr}_{weighted_str}/seed_{args.seed}"
    if model_load_dir == '' or args.uniform:
        results_save_dir = f"./exps/{args.folder}/results/{train_type}/traindata_{train_data}/testdata_{test_data}/{args.i3d_training}/{ssl_str}/{num_stages}_{num_layers}_{num_f_maps}_{features_dim}_{bz}_{lr}_{weighted_str}/seed_{args.seed}/th_{args.classification_threshold}"
    else:
        results_save_dir = model_save_dir.replace('models', 'results').replace(f'traindata_{train_data}', f'traindata_{train_data}/testdata_{test_data}')
        results_save_dir = f'{results_save_dir}/th_{args.classification_threshold}'

    if args.action == 'train':
        if os.path.exists(model_save_dir) and not args.refresh:
            print(f'Model directory already exists: {model_save_dir}')
            sys.exit()
        else:
            os.makedirs(model_save_dir, exist_ok=True)
            save_args(args, model_save_dir)

    elif args.extract_save_pseudo_labels == 0:
        if os.path.exists(results_save_dir):
            print(f'Results directory already exists : {results_save_dir}')
            sys.exit()
        else:
            os.makedirs(results_save_dir)
            save_args(args, results_save_dir)
            
    return model_load_dir, model_save_dir, results_save_dir