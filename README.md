# Temporal segmentation of sign language videos
This repository provides code for following two papers:

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/) and [Gül Varol](https://www.robots.ox.ac.uk/~gul),
*Sign language segmentation with temporal convolutional networks*, ICASSP 2021.  [[arXiv]](https://arxiv.org/abs/2011.12986)

- [Katrin Renz](https://www.katrinrenz.de), [Nicolaj C. Stache](https://www.hs-heilbronn.de/nicolaj.stache), [Neil Fox](https://www.ucl.ac.uk/dcal/people/research-staff/neil-fox), [Gül Varol](https://www.robots.ox.ac.uk/~gul) and [Samuel Albanie](https://www.robots.ox.ac.uk/~albanie/),
*Sign Segmentation with Changepoint-Modulated Pseudo-Labelling*, CVPRW 2021. [[arXiv]](https://arxiv.org/abs/2104.13817)

[[Project page]](https://www.robots.ox.ac.uk/~vgg/research/signsegmentation/)

![demo](demo/results/demo.gif)

## Contents
* [Setup](#setup)
* [Data and models](#data-and-models)
* [Demo](#demo)
* [Training](#training)
  * [Train ICASSP](#train-icassp)
  * [Train CVPRW](#train-cvprw)
* [Citation](#citation)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Setup

``` bash
# Clone this repository
git clone git@github.com:RenzKa/sign-segmentation.git
cd sign-segmentation/
# Create signseg_env environment
conda env create -f environment.yml
conda activate signseg_env
```

## Data and models
You can download our pretrained models (`models.zip [302MB]`) and data (`data.zip [5.5GB]`) used in the experiments [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing) or by executing `download/download_*.sh`. The unzipped `data/` and `models/` folders should be located on the root directory of the repository (for using the demo downloading the `models` folder is sufficient).


### Data:
Please cite the original datasets when using the data: [BSL Corpus](https://bslcorpusproject.org/cava/acknowledgements-and-citation/) | [Phoenix14](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
We provide the pre-extracted features and metadata. See [here](data/README.md) for a detailed description of the data files. 
- Features: `data/features/*/*/features.mat`
- Metadata: `data/info/*/info.pkl`

### Models:
- I3D weights, trained for sign classification: `models/i3d/*.pth.tar`
- MS-TCN weights for the demo (see tables below for links to the other models): `models/ms-tcn/*.model`

The folder structure should be as below:
```
sign-segmentation/models/
  i3d/
    i3d_kinetics_bsl1k_bslcp.pth.tar
    i3d_kinetics_bslcp.pth.tar
    i3d_kinetics_phoenix_1297.pth.tar
  ms-tcn/
    mstcn_bslcp_i3d_bslcp.model
```
## Demo
The demo folder contains a sample script to estimate the segments of a given sign language video. It is also possible to use pre-extracted I3D features as a starting point, and only apply the MS-TCN model.
`--generate_vtt` generates a `.vtt` file which can be used with [our modified version of VIA annotation tool](https://github.com/RenzKa/VIA_sign-language-annotation):
```
usage: demo.py [-h] [--starting_point {video,feature}]
               [--i3d_checkpoint_path I3D_CHECKPOINT_PATH]
               [--mstcn_checkpoint_path MSTCN_CHECKPOINT_PATH]
               [--video_path VIDEO_PATH] [--feature_path FEATURE_PATH]
               [--save_path SAVE_PATH] [--num_in_frames NUM_IN_FRAMES]
               [--stride STRIDE] [--batch_size BATCH_SIZE] [--fps FPS]
               [--num_classes NUM_CLASSES] [--slowdown_factor SLOWDOWN_FACTOR]
               [--save_features] [--save_segments] [--viz] [--generate_vtt]
```

Example usage:
``` bash
# Print arguments
python demo/demo.py -h
# Save features and predictions and create visualization of results in full speed
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 1 --save_features --save_segments --viz
# Save only predictions and create visualization of results slowed down by factor 6
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --save_segments --viz
# Create visualization of results slowed down by factor 6 and .vtt file for VIA tool
python demo/demo.py --video_path demo/sample_data/demo_video.mp4 --slowdown_factor 6 --viz --generate_vtt
```

The demo will: 
1. use the `models/i3d/i3d_kinetics_bslcp.pth.tar` pretrained I3D model to extract features,
2. use the `models/ms-tcn/mstcn_bslcp_i3d_bslcp.model` pretrained MS-TCN model to predict the segments out of the features,
3. save results (depending on which flags are used).

## Training
### Train ICASSP
Run the corresponding run-file (`*.sh`) to train the MS-TCN with pre-extracted features on BSL Corpus.
During the training a `.log` file for tensorboard is generated. In addition the metrics get saved in `train_progress.txt`.

* Influence of I3D training (fully-supervised segmentation results on BSL Corpus)

|ID | Model | mF1B | mF1S | Links (for seed=0) |
|   -   |   -  |   -  |   -   |  -   | 
| 1 | BSL Corpus | 68.68<sub>±0.6</sub> |47.71<sub>±0.8</sub> | [run](https://drive.google.com/file/d/1na-b_WoPPajPN9WCd8kdQ0WrMKr-EBzH/view?usp=sharing), [args](https://drive.google.com/file/d/1FHC0mHt3meXBobnuPWN17vwlPaekF-qU/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1ot6VNYfzn9UlVdRt31mQ8Mfics1DxP7T/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1y-LeaNuZSAeLTc1yKXA0XUWnwINr8pVo?usp=sharing) |
| 2 | BSL1K -> BSL Corpus | 66.17<sub>±0.5</sub> |44.44<sub>±1.0</sub> | [run](https://drive.google.com/file/d/1hvRN7a3GX7YQF9jTxfmsJbS_WXebsb9Y/view?usp=sharing), [args](https://drive.google.com/file/d/1Gg_qZYYUtl3YtNQec2ku-VxBiM9r-0mR/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1ETNO6tLgmg_o-T7L0qG4eMEd8QhsZpOG/view?usp=sharing), [logs](https://drive.google.com/drive/folders/11WnAEmYY3PIC03XdBNZB0lU4BkZmVqKp?usp=sharing) |


* Fully-supervised segmentation results on PHOENIX14

|ID | I3D training data | MS-TCN training data | mF1B | mF1S | Links (for seed=0) |
| - |   -   |   -  |   -  |   -   |   -   | 
|3| BSL Corpus | PHOENIX14 | 65.06<sub>±0.5</sub> |44.42<sub>±2.0</sub> | [run](https://drive.google.com/file/d/1Vihh4MG0iWOLQalI5SqVYjQn3aELsLRP/view?usp=sharing), [args](https://drive.google.com/file/d/1PLN7wcsJBqnhIBWkdfgPvyujh3MIzea4/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1hynccDWvwKaH8uiMRAVYiWsqeK9dptm5/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1Rfklvh3-pdCe_meKOcw9rjLCR_R5j-ap?usp=sharing) |
|4| PHOENIX14 | PHOENIX14 | 71.50<sub>±0.2</sub> |52.78<sub>±1.6</sub> | [run](https://drive.google.com/file/d/1jAfJPs58ErT-UTnN3mPstOhekAgTCLEY/view?usp=sharing), [args](https://drive.google.com/file/d/1ak6VOuLvv6hrDEUsJbI1WcGoJGfVpUi5/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1q0bShF9IpuuSHrJyZIPNMUsC5guQ0B8m/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1wk5dly6jxKEivO3q5BEahTtLSsrPK1of?usp=sharing) |



### Train CVPRW
*Requirement:* pre-extracted pseudo-labels/ changepoints or CMPL-labels:
1. Save pre-trained model in ```models/ms-tcn/*.model```
2. a) Extract pseudo-labels before extracting CMPL-labels: [Extract only PL](https://drive.google.com/file/d/1ixM0leAeR_JU2n-yZuo6O7zA1h_amUBR/view?usp=sharing) | [Extract CMPL](https://drive.google.com/file/d/1juiYMWX_V8Au8LAoPYycgD5ejYA8ZlZg/view?usp=sharing) | [Extract PL and CMPL](https://drive.google.com/file/d/1lvbAgqCdmxsPcAZysGJVGTsjr7IgCplj/view?usp=sharing)
 b) Extract Changepoints separately for training: [Extract CP](https://drive.google.com/file/d/1urZxzZ4qlK46Ad5_8SJDzXVu4NFlGbNn/view?usp=sharing)
-> specify correct model path

* Pseudo-labelling techniques on PHOENIX14

|ID| Method | Adaptation protocol | mF1B | mF1S | Links (for seed=0) |
|-|   -   |   -  |   -  |   -   |   -   | 
|5| Pseudo-labels | inductive | 47.94<sub>±1.0</sub> | 32.45<sub>±0.3</sub> | [run](https://drive.google.com/file/d/1qgvJsKzgA-eutZVAba_jt2TucXYLTnDM/view?usp=sharing), [args](https://drive.google.com/file/d/1dcC-heRxyGnnAVgKFvprkhRbrPCkqPMn/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1pqDmCo_GBvWx0OO2mRbQdoy_9lG-TDEo/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1xSYTd7mCqmmk6pCGrHAncS5RoxJjwQ0-?usp=sharing) |
|6| Changepoints | inductive | 48.51<sub>±0.4</sub> | 34.45<sub>±1.4</sub> | [run](https://drive.google.com/file/d/117HicDNam1sMFI7uV4dzOW-J8RoMZL63/view?usp=sharing), [args](https://drive.google.com/file/d/1skuhL7tP9BhuaZ6Vh9u_BO3AJAmmQmkD/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1LP5alfee4DfGZVz4bkWO7v2GoJ7OP4cD/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1jC5w7Dg5LKse4T4OtTCYM_0w7JCW5OnF?usp=sharing) |
|7| CMPL | inductive | 53.57<sub>±0.7</sub> | 33.82<sub>±0.0</sub> | [run](https://drive.google.com/file/d/1lbhiQNqYpPWbxRmrxtRt-R-CzEbw21Ng/view?usp=sharing), [args](https://drive.google.com/file/d/16iQj5e0X8za0pMPZnFzst7el1jSUqDtk/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/13SM0Lh4_89am9RJvwc-3uDLbV_DwA-W0/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1B4j4H2arehZjDNL2bjnM2I7LtAHHmm43?usp=sharing) |
|8| Pseudo-labels | transductive | 47.62<sub>±0.4</sub> | 32.11<sub>±0.9</sub> | [run](https://drive.google.com/file/d/1HrWZ0LM_OR9bQL7ZKRVM_LAzb13d1QWX/view?usp=sharing), [args](https://drive.google.com/file/d/1KzuA_iqMtJ94Vlm6jRpslwxVgNmVSuvX/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1TU1gf1AA1eGi5PBcBqvA_04Fynxu4xbx/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1NdBzs9l4KIZ05g0-sKi2ftwvtZlhTl9a?usp=sharing) |
|9| Changepoints | transductive | 48.29<sub>±0.1</sub> | 35.31<sub>±1.4</sub> | [run](https://drive.google.com/file/d/1fuqwrGyCjamUCY7DU2MMWnMPGeFVE6KU/view?usp=sharing), [args](https://drive.google.com/file/d/1h_YrYK9Q7F8shbMMkeaWbk1VrgJYmXXT/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/1N3qQlrvVOp79pAJ9uIo_fVTYMN_YXLL7/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1e83LQ7ZbWdj6t9OYNsn7RRHiljm0zJF_?usp=sharing) |
|10| CMPL | transductive | 53.53<sub>±0.1</sub> | 32.93<sub>±0.9</sub> | [run](https://drive.google.com/file/d/1_Ql5aia7Jo2xszeXZk7tbLP00z7bqrht/view?usp=sharing), [args](https://drive.google.com/file/d/1T3I2V53aPmdHHFZH4PDbo9pPCz8JIg1p/view?usp=sharing), [I3D model](https://drive.google.com/file/d/1pbldZLqFSS2E0hoh1f4hiTvB9jL3QGOB/view?usp=sharing), [MS-TCN model](https://drive.google.com/file/d/10tGdIcJCrJ9n40apjWXTN6KcQy0SUsNx/view?usp=sharing), [logs](https://drive.google.com/drive/folders/1xz_BSamWmxyBseXl50-2vh-2cUfTUkMy?usp=sharing) |

## Citation
If you use this code and data, please cite the following:

```
@inproceedings{Renz2021signsegmentation_a,
    author       = "Katrin Renz and Nicolaj C. Stache and Samuel Albanie and G{\"u}l Varol",
    title        = "Sign Language Segmentation with Temporal Convolutional Networks",
    booktitle    = "ICASSP",
    year         = "2021",
}
```
```
@inproceedings{Renz2021signsegmentation_b,
    author       = "Katrin Renz and Nicolaj C. Stache and Neil Fox and G{\"u}l Varol and Samuel Albanie",
    title        = "Sign Segmentation with Changepoint-Modulated Pseudo-Labelling",
    booktitle    = "CVPRW",
    year         = "2021",
}
```

## License
The license in this repository only covers the code. For data.zip and models.zip we refer to the terms of conditions of original datasets.


## Acknowledgements
The code builds on the [github.com/yabufarha/ms-tcn](https://github.com/yabufarha/ms-tcn) repository. The demo reuses parts from [github.com/gulvarol/bsl1k](https://github.com/gulvarol/bsl1k).  We like to thank C. Camgoz for the help with the BSLCORPUS data preparation.