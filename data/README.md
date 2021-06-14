# Sign segmentation data documentation

Data and models can be downloaded [here](https://drive.google.com/drive/folders/17DaatdfD4GRnLJJ0RX5TcSfHGMxMS0Lm?usp=sharing).

* `features.mat` contains 1024 dimensional I3D features.

* Components of the `info.pkl` file:
``` bash
{
    ["words"]                   # word vocabulary 
    ["words_to_id"]             # dictionary from word => a unique word id 
    # Dictionary containing metadata for each of the videos
    # each key represents a list of values
    ["videos"]{
        name                    # our unique naming, filename of the .mp4 clip
        org_name                # path of the original video (full length)
        start                   # start time of the clip in the original video
        end                     # end time of the clip in the original video
        signer                  # abbreviation of region and signer id
        split                   # assignment to split, 0: train, 1: eval, 2: test
        glosses                 # sentence-level, gloss names
        gloss_ids               # sentence-level, gloss id
        
        # Frame-level annotations
        ["alignments"]{
            boundaries              # frame-level, 0: sign, 1: boundary
            gloss                   # frame-level, gloss name
            gloss_id                # frame-level, gloss id
        }
        # Dictionary containing video resolution information for the original videos
        ["videos"]{
            T                   # number of frames
            H                   # height
            W                   # width
            duration_sec        # seconds
            fps                 # frames per second
        }
    }
}
```
