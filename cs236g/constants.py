"""Constants relevant to preprocessing and/or the model"""
### constants
FILEPATH = "drive/MyDrive/big-data-bowl-2021/"

### the (basic) personnel groupings we'll consider
OFF_PERSONNEL_TYPES = [
    "1 RB, 1 TE, 3 WR",
    "1 RB, 2 TE, 2 WR",
    "2 RB, 1 TE, 2 WR",
    "1 RB, 3 TE, 1 WR",
    "1 RB, 0 TE, 4 WR",
    "0 RB, 1 TE, 4 WR",
    "2 RB, 2 TE, 1 WR",
    "2 RB, 0 TE, 3 WR",
    "0 RB, 2 TE, 3 WR",
    "0 RB, 0 TE, 5 WR",
]

### number of frames in each tensor
N_FRAMES = 75
### dimension, i.e. width, of each tensor
FRAME_WIDTH = 25

### approximate min-max scaling values. Min presumed 0.
SCALE_VALUES = {"X": 55.0, "Y": 120.0, "SPD": 12.0, "ACC": 22.0}

### initial dims of latent space
Z_DIM = 8
