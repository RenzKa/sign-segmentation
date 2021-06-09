#!/usr/bin/python3.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
from pathlib import Path
import torch

from utils.utils import get_labels_start_end_time, torch_to_list


class Metric():
    def __init__(self, run_type):
        self.run_type = run_type
        self.overlap = list(np.arange(0.4, 0.76, 0.05))
        self.thresholds_b = list(range(1, 5))

        self.correct = 0
        self.total = 0
        self.width_gt = []
        self.width_pred = []

        self.iou = []
        self.num_det, self.num_gt = 0, 0
        self.dist = []
        self.tp_list = np.zeros(len(self.overlap))
        self.fp_list = np.zeros(len(self.overlap))
        self.fn_list = np.zeros(len(self.overlap))

        self.tp_list_b = np.zeros(len(self.thresholds_b))
        self.fp_list_b = np.zeros(len(self.thresholds_b))
        self.fn_list_b = np.zeros(len(self.thresholds_b))

        self.tp_listm = np.zeros(len(self.overlap))
        self.fp_listm = np.zeros(len(self.overlap))
        self.fn_listm = np.zeros(len(self.overlap))

    def calc_scores_per_batch(self, pred, gt, gt_eval, mask=None):
        if mask is None:
            self.correct += (pred == gt_eval).int().sum().item()
            self.total += gt_eval.shape[1]
        else:
            self.correct += ((pred == gt).float()*mask[:, 0, :].squeeze(1)).sum().item()
            self.total += torch.sum(mask[:, 0, :]).item()

        pred = [item for sublist in torch_to_list(pred) for item in sublist]
        gt_eval = [item for sublist in torch_to_list(gt_eval) for item in sublist]
        gt = [item for sublist in torch_to_list(gt) for item in sublist]

        tp_list1, fp_list1, fn_list1, num_det1, num_gt1, dist1, width_pred1, width_gt1 = get_boundary_metric(pred, gt_eval, self.thresholds_b, bg_class=[0, -100])
        self.tp_list_b += tp_list1
        self.fp_list_b += fp_list1
        self.fn_list_b += fn_list1
        self.num_det += num_det1
        self.num_gt += num_gt1
        self.dist.extend(dist1)
        self.width_gt.append(width_gt1)
        self.width_pred.append(width_pred1)

        tp_list1, fp_list1, fn_list1, mean_iou = get_sign_metric(pred, gt, self.overlap, bg_class=[1, -100])
        self.iou.append(mean_iou)
        self.tp_list += tp_list1
        self.fp_list += fp_list1
        self.fn_list += fn_list1

    def calc_metrics(self):
        self.mean_iou_ges = np.mean(self.iou)
        self.f1_sign = []
        for s in range(len(self.overlap)):
            self.precision = self.tp_list[s] / float(self.tp_list[s]+self.fp_list[s])
            self.recall = self.tp_list[s] / float(self.tp_list[s]+self.fn_list[s])

            f1 = 2.0 * (self.precision*self.recall) / (self.precision+self.recall)

            f1 = np.nan_to_num(f1)*100
            self.f1_sign.append(round(f1, 2))

        self.mean_f1s = np.mean(self.f1_sign)
        self.f1_sign = [self.f1_sign[2], self.f1_sign[-1]]

        self.f1b = []
        self.recall_list = []
        self.precision_list = []
        for s in range(len(self.thresholds_b)):
            self.precision_b = self.tp_list_b[s] / float(self.tp_list_b[s]+self.fp_list_b[s])
            self.recall_b = self.tp_list_b[s] / float(self.tp_list_b[s]+self.fn_list_b[s])

            f1_b = 2.0 * (self.precision_b*self.recall_b) / (self.precision_b+self.recall_b)

            f1_b = np.nan_to_num(f1_b)*100
            self.f1b.append(f1_b)
            self.recall_list.append(self.recall_b*100)
            self.precision_list.append(self.precision_b*100)

        self.mean_f1b = round(np.mean(self.f1b), 2)
        self.mean_recall_b = round(np.mean(self.recall_list), 2)
        self.mean_precision_b = round(np.mean(self.precision_list), 2)

        self.mean_dist = np.mean(np.abs(self.dist))
        self.mean_width_pred = np.mean(self.width_pred)
        self.mean_width_gt = np.mean(self.width_gt)


    def save_print_metrics(self, writer, save_dir, epoch, epoch_loss):

        writer.add_scalar(f'{self.run_type}/mF1B', self.mean_f1b, epoch+1)
        writer.add_scalar(f'{self.run_type}/mF1S', self.mean_f1s, epoch+1)

        #add to dict
        result_dict = {epoch: {}}
        result_dict[epoch] = {
            'mF1B': self.mean_f1b,
            'F1S': self.f1_sign[-1],
            'mF1S': self.mean_f1s,
            'IoU': 100*self.mean_iou_ges,
            'widthB': self.mean_width_pred,
            'dist': self.mean_dist,
            'detB': self.num_det,
        }

        if self.run_type == 'train' or self.run_type == 'eval':
            print_str = f"[E{epoch + 1} / {self.run_type}]: epoch loss = {epoch_loss:.4f},   acc = {100*float(self.correct)/self.total:.2f}, mean F1B = {self.mean_f1b:.2f}, mean_F1S = {self.mean_f1s:.2f}"
            save_str = f"[E{epoch + 1} / {self.run_type}]: epoch loss = {epoch_loss:.4f},   acc = {100*float(self.correct)/self.total:.2f}, F1_bound = {self.f1b}, mean_recall_b = {self.mean_recall_b}, mean_precision_b = {self.mean_precision_b}, mean F1B = {self.mean_f1b:.2f}, F1_sign = {self.f1_sign}, mean_F1S = {self.mean_f1s:.2f}, IoU_sign = {100*self.mean_iou_ges:.2f}, #B ({self.num_det}/{self.num_gt}), dist = {self.mean_dist}, boundary width: {self.mean_width_pred:.2f} / {self.mean_width_gt:.2f}  \n"

            if self.run_type == 'train':
                print(f'\n{print_str}')
            else:
                print(f'{print_str}\n')

            with open(f'{save_dir}/train_progress.txt', 'a+') as f:
                if self.run_type == 'train':
                    f.write('\n\n ---------------------------------------------------\n')
                f.write(save_str)

        elif self.run_type == 'test':
            save_dir = f'{str(Path(save_dir).parent)}/{str(Path(save_dir).stem)}.txt'

            print(f"Acc: {(100*float(self.correct)/self.total):.2f}")
            print(f"F1B: {(self.f1b)}")
            print(f"mean Recall B: {(self.mean_recall_b)}")
            print(f"mean Precision B: {(self.mean_precision_b)}")
            print(f'mean F1B: {(self.mean_f1b):.2f}')
            print(f'IoU Sign: {(100*self.mean_iou_ges):.2f}')
            print(f'F1s: {(self.f1_sign)}')
            print(f'mean F1S: {(self.mean_f1s):.2f}')
            print(f'#B:  ({self.num_det}/{self.num_gt})')
            print(f'mean dist [frames]: {self.mean_dist:.2f}')
            print(f'mean B width pred/gt [frames]: {self.mean_width_pred:.2f} / {self.mean_width_gt:.2f} \n')

            with open(save_dir, 'a+') as f:
                f.write(f'Acc: {(100*float(self.correct)/self.total):.2f} \n')
                f.write(f'F1b: {(self.f1b)} \n')
                f.write(f'mean Recall B: {(self.mean_recall_b):.2f} \n')
                f.write(f'mean Precision B: {(self.mean_precision_b):.2f} \n')
                f.write(f'mean F1b: {(self.mean_f1b):.2f} \n')
                f.write(f'IoU Sign: {(100*self.mean_iou_ges):.2f} \n')
                f.write(f'F1s: {(self.f1_sign)} \n')
                f.write(f'mean F1S: {(self.mean_f1s):.2f} \n')
                f.write(f'#B:  ({self.num_det}/{self.num_gt})  \n')
                f.write(f'mean dist [frames]: {self.mean_dist:.2f} \n')
                f.write(f'mean B width pred/gt [frames]: {self.mean_width_pred:.2f} / {self.mean_width_gt:.2f} \n')

        return result_dict


def get_boundary_metric(pred, gt, thresholds, bg_class=[0]):

    p_label, p_start, p_end = get_labels_start_end_time(pred, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(gt, bg_class)

    mean_boundary_width_pred = list(pred).count(1) / len(p_label) if len(p_label) else 0
    mean_boundary_width_gt = list(gt).count(1) / len(y_label) if len(y_label) else 0

    num_pred = len(p_label)
    num_gt = len(y_label)

    pos_p = [(p_end[i]+p_start[i])/2 for i in range(len(p_label))]
    pos_y = [(y_end[i]+y_start[i])/2 for i in range(len(y_label))]

    # calculate distance matrix
    if len(p_label) > 0:
        dist_all = []
        for p in pos_p:
            dist_all.append([abs(y-p) for y in pos_y])
        dist_arr = np.asarray(dist_all)

        # calculate mean distance
        mean_dist = [np.mean(np.min(dist_arr, 1))]

    else:
        mean_dist = [0]

    # find smallest distances
    dist_choosen = []
    if len(p_label) > 0:
        for ix in range(min(dist_arr.shape[0], dist_arr.shape[1])):

            argmin_row = np.argmin(dist_arr, axis=1)
            min_row = np.min(dist_arr, axis=1)
            min_dist = np.min(min_row)
            argmin_dist = np.argmin(min_row)

            dist_choosen.append(min_dist)

            # delete row and column -> pred-gt pair can't be reused
            dist_arr = np.delete(dist_arr, argmin_dist, 0)
            dist_arr = np.delete(dist_arr, argmin_row[argmin_dist], 1)

    tp_list = []
    fp_list = []
    fn_list = []

    for th in thresholds:
        tp = 0
        fp = 0
        for dist in dist_choosen:
            if dist <= th:
                tp += 1
            else:
                fp += 1

        # more predictions than gt -> count as false positiv
        fp += max(0, len(p_label)-len(dist_choosen))
        # difference between number of true boundaries and correct predicted ones -> false negative
        fn = len(y_label) - tp

        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    return np.asarray(tp_list), np.asarray(fp_list), np.asarray(fn_list), num_pred, num_gt, mean_dist, mean_boundary_width_pred, mean_boundary_width_gt


def get_sign_metric(pred, gt, overlap, bg_class=[1]):
    p_label, p_start, p_end = get_labels_start_end_time(pred, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(gt, bg_class)
    iou_all = []

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        iou_all.append((1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))]))

    iou_arr = np.asarray(iou_all)
    iou_choosen = []
    if len(p_label) > 0:
        for ix in range(min(iou_arr.shape[0], iou_arr.shape[1])):
            argmax_row = np.argmax(iou_arr, axis=1)
            max_row = np.max(iou_arr, axis=1)
            max_iou = np.max(max_row)
            argmax_iou = np.argmax(max_row)
            iou_choosen.append(max_iou)
            iou_arr = np.delete(iou_arr, argmax_iou, 0)
            iou_arr = np.delete(iou_arr, argmax_row[argmax_iou], 1)

        diff = max(iou_arr.shape[0], iou_arr.shape[1]) - len(iou_choosen)
    else:
        diff = len(y_label)

    tp_list = []
    fp_list = []
    fn_list = []

    for ol in overlap:
        tp = 0
        fp = 0
        for match in iou_choosen:
            if match > ol:
                tp += 1
            else:
                fp += 1
        fp += max(0, len(p_label)-len(iou_choosen))
        fn = len(y_label) - tp
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    iou_choosen.extend([0]*diff)
    mean_iou = np.mean(iou_choosen)

    return np.asarray(tp_list), np.asarray(fp_list), np.asarray(fn_list), mean_iou
