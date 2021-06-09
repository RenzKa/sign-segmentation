import numpy as np
import torch
import cv2
import scipy.misc
import scipy.ndimage
import scipy.io
import os
import pickle
from math import floor
import datetime
from webvtt import WebVTT, Caption
from PIL import Image, ImageDraw, ImageFont

def torch_to_list(torch_tensor):
    return torch_tensor.cpu().numpy().tolist()

def save_pred(preds, checkpoint="checkpoint", filename="preds_valid.mat"):
    preds = to_numpy(preds)
    checkpoint.mkdir(exist_ok=True, parents=True)
    filepath = os.path.join(checkpoint, filename)
    mdict = {"preds": preds}
    print(f"Saving to {filepath}")
    scipy.io.savemat(filepath, mdict=mdict, do_compression=False, format="4")

def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError(f"Cannot convert {type(ndarray)} to torch tensor")
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
    return tensor

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def resize_generic(img, oheight, owidth, interp="bilinear", is_flow=False):
    """
    Args
    inp: numpy array: RGB image (H, W, 3) | video with 3*nframes (H, W, 3*nframes)
          |  single channel image (H, W, 1) | -- not supported:  video with (nframes, 3, H, W)
    """

    # resized_image = cv2.resize(image, (100, 50))
    ht, wd, chn = img.shape[0], img.shape[1], img.shape[2]
    if chn == 1:
        resized_img = scipy.misc.imresize(
            img.squeeze(), [oheight, owidth], interp=interp, mode="F"
        ).reshape((oheight, owidth, chn))
    elif chn == 3:
        # resized_img = scipy.misc.imresize(img, [oheight, owidth], interp=interp)  # mode='F' gives an error for 3 channels
        resized_img = cv2.resize(img, (owidth, oheight))  # inverted compared to scipy
    elif chn == 2:
        # assert(is_flow)
        resized_img = np.zeros((oheight, owidth, chn), dtype=img.dtype)
        for t in range(chn):
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp)
            # resized_img[:, :, t] = scipy.misc.imresize(img[:, :, t], [oheight, owidth], interp=interp, mode='F')
            # resized_img[:, :, t] = np.array(Image.fromarray(img[:, :, t]).resize([oheight, owidth]))
            resized_img[:, :, t] = scipy.ndimage.interpolation.zoom(
                img[:, :, t], [oheight, owidth]
            )
    else:
        in_chn = 3
        # Workaround, would be better to pass #frames
        if chn == 16:
            in_chn = 1
        if chn == 32:
            in_chn = 2
        nframes = int(chn / in_chn)
        img = img.reshape(img.shape[0], img.shape[1], in_chn, nframes)
        resized_img = np.zeros((oheight, owidth, in_chn, nframes), dtype=img.dtype)
        for t in range(nframes):
            frame = img[:, :, :, t]  # img[:, :, t*3:t*3+3]
            frame = cv2.resize(frame, (owidth, oheight)).reshape(
                oheight, owidth, in_chn
            )
            # frame = scipy.misc.imresize(frame, [oheight, owidth], interp=interp)
            resized_img[:, :, :, t] = frame
        resized_img = resized_img.reshape(
            resized_img.shape[0], resized_img.shape[1], chn
        )

    if is_flow:
        # print(oheight / ht)
        # print(owidth / wd)
        resized_img = resized_img * oheight / ht
    return resized_img


def color_normalize(x, mean, std):
    """Normalize a tensor of images by subtracting (resp. dividing) by the mean (resp.
    std. deviation) statistics of a dataset in RGB space.
    """
    if x.dim() in {3, 4}:
        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)
        assert x.size(0) == 3, "For single video format, expected RGB along first dim"
        for t, m, s in zip(x, mean, std):
            t.sub_(m)
            t.div_(s)
    elif x.dim() == 5:
        assert (
            x.shape[1] == 3
        ), "For batched video format, expected RGB along second dim"
        x[:, 0].sub_(mean[0]).div_(std[0])
        x[:, 1].sub_(mean[1]).div_(std[1])
        x[:, 2].sub_(mean[2]).div_(std[2])
    return x

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

def generate_vtt_file(all_preds, logits, save_path):
    vtt = WebVTT()
    predictions = all_preds

    labels, starts, ends = get_labels_start_end_time(predictions, [1])

    # smaller boundaries
    for ix in range(len(labels)):
        if ix == len(labels)-1:
            break
        diff = starts[ix+1]-ends[ix]
        starts[ix+1] -= floor(diff/2)
        ends[ix] += floor(diff/2)

    # load i3d classes
    i3d_scores = logits
    with open('data/info/bslcp/info.pkl', 'rb') as f:
        info_data = pickle.load(f)

    # for start, end in zip(starts, ends):
    for start, end in zip(starts, ends):

        if logits is not None:
            i3d_score = np.sum(np.asarray(i3d_scores)[start:end], axis=0)
            ind = np.argpartition(i3d_score, -10)[-10:]       
            ind = ind[np.argsort(-i3d_score[ind])]
            classes = [info_data['words'][ix] for ix in ind]

            class_str = ','.join(classes)
        else:
            class_str = ''

        start = (start + 8) / 25
        end = (end + 8) / 25

        start_dt = datetime.timedelta(seconds=start)
        start_str = str(start_dt)
        if '.' not in start_str:
            start_str = f'{start_str}.000000'

        end_dt = datetime.timedelta(seconds=end)
        end_str = str(end_dt)
        if '.' not in end_str:
            end_str = f'{end_str}.000000'
        # creating a caption with a list of lines
        caption = Caption(
            start_str,
            end_str,
            [class_str]
        )

        # adding a caption
        vtt.captions.append(caption)


    # save to a different file
    vtt.save(f'{save_path}/demo.vtt')

    
