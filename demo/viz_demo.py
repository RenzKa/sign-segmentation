import os
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def video_results(video_path, pred, pred_prob, save_path, slowdown_factor, num_in_frames=16):
    
    pre_pad, post_pad = 8, 7
    visible_window = 80
    offset = int(visible_window // 2)

    pred = [0]*(offset+pre_pad) + pred + [0]*(offset+post_pad)
    pred_prob = [0]*(offset+pre_pad) + pred_prob + [0]*(offset+post_pad)


    palette = [(0.9,0.9,0.9), sns.color_palette("colorblind", 10)[0]]   # deep, muted, pastel, bright, dark, colorblind
    palette2 = [(0.9,0.9,0.9), sns.color_palette("pastel", 10)[0]]
    save_path_video = f'{save_path}/{video_path.stem}_result.mp4'

    cap = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_frames = len(pred) - visible_window
    msg = f"Expected {expected_frames} frames, found {num_frames}"
    assert expected_frames == num_frames, msg

    writer = cv2.VideoWriter(
            save_path_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (width, height),
    )

    # frame width in pixels for viz
    fw_px = int((512-73-12)/len(pred))

    for idx in range(num_frames):
        seg_idx = 1 # idx + pre_pad + offset
        ret, frame = cap.read()
        if not ret:
            break

        plt.close("all")
        plt.clf()

        if seg_idx>= 0:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            boundaries_ = np.array(pred[idx:idx + visible_window])
            probs_ = np.array(pred_prob[idx:idx + visible_window])
            fig = plt.figure(figsize=(width/100, height/100), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(frame, aspect="equal", extent=None)
            plt.xticks([])
            plt.yticks([])
            ax.text(0.2*width, height - 0.08*height, "preds", bbox={'facecolor': 'white', "alpha": 0.5})

            y_scale = 25
            x_scale = 2
            xticks = np.arange(len(boundaries_)) * x_scale

            # ax.invert_yaxis()
            ax = plt.axes([0.25, 0.05, 0.5, 0.08], facecolor='w')
            ax.bar(xticks, y_scale * boundaries_, width=5, color=palette[1])
            ax.bar(xticks, y_scale * (1 - boundaries_), width=5, color=palette2[0])
            ax.plot(xticks, y_scale * probs_, color="black", lw=2)
            ax.axvline(xticks[offset], ymin=0, ymax=1, lw=3, color="green")
            ax.axis("off")

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        else:
            img = frame
        for k in range(slowdown_factor):
            writer.write(img)

    cap.release()
    writer.release()