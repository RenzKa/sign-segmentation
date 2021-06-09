import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, gt_dict, features_dict, gt_dict_eval=None):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.gt_dict = gt_dict
        self.features_dict = features_dict
        self.gt_dict_eval = gt_dict_eval

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def get_max_index(self):
        return len(self.list_of_examples)

    def read_data(self, vid_list_file):
        self.list_of_examples = vid_list_file
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_eval = []
        for vid in batch:
            features = np.swapaxes(self.features_dict[vid], 0, 1)
            classes = self.gt_dict[vid]

            batch_input.append(features)
            batch_target.append(np.asarray(classes))

            if self.gt_dict_eval is not None:
                classes_eval = self.gt_dict_eval[vid]
                batch_target_eval.append(np.asarray(classes_eval))

        shape_index = 0
        length_of_sequences = [item.shape[shape_index] for item in batch_target]
        batch_input_tensor = torch.zeros(len(batch), np.shape(batch_input[0])[0],  max(length_of_sequences), dtype=torch.float)

        # regression
        if self.num_classes == 1:
            batch_target_tensor = torch.ones(len(batch), max(length_of_sequences), dtype=torch.float)*(-100.)
            batch_target_eval_tensor = torch.ones(len(batch), max(length_of_sequences), dtype=torch.float)*(-100.)
            shape_index = 0

        # classification
        else:
            batch_target_tensor = torch.ones(len(batch), max(length_of_sequences), dtype=torch.long)*(-100)
            batch_target_eval_tensor = torch.ones(len(batch), max(length_of_sequences), dtype=torch.long)*(-100)
            shape_index = 0

        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[shape_index]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[shape_index]] = torch.ones(self.num_classes, np.shape(batch_target[i])[shape_index])
            if self.gt_dict_eval is not None:
                batch_target_eval_tensor[i, :np.shape(batch_target_eval[i])[shape_index]] = torch.from_numpy(batch_target_eval[i])

        if self.gt_dict_eval is not None:
            return batch_input_tensor, batch_target_tensor, batch_target_eval_tensor, mask
        else:
            return batch_input_tensor, batch_target_tensor, mask
