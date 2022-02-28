"""Constants relevant to preprocessing and/or the model"""
from collections import namedtuple

### constants
FILEPATH = "drive/MyDrive/CS236G/data/"
RUNPATH = "drive/MyDrive/CS236G/runs/"

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

### tagged route names
ROUTE_LABELS = [
    "ANGLE",
    "CORNER",
    "CROSS",
    "FLAT",
    "GO",
    "HITCH",
    "IN",
    "OUT",
    "POST",
    "SCREEN",
    "SLANT",
    "WHEEL",
    "undefined",
]

### follow these over the course of a play, per player
PLAYER_TENSOR_COLS = ["x_coord", "y_coord", "s", "a"]

### number of frames in each tensor
N_FRAMES = 75
### dimension, i.e. width, of each tensor
FRAME_WIDTH = 25

### approximate min-max scaling values. Min presumed 0. unless presented as list
SCALE_VALUES = {
    "X": 55.0,
    "Y": 140.0,
    "Y_LOS": [-15.0, 40.0],
    "SPD": 1.0,  # 12.0,
    "ACC": 1.0,  # 24.0
}

### initial dims of latent space
Z_DIM = 24
### dimension of play start
Z_SUPP_DIM = 89


### use this to store model configuration
Configs = namedtuple(
    "Configs",
    [
        "n_epochs",
        "z_dim",
        "display_step",
        "save_step",
        "batch_size",
        "lr",
        "beta_1",
        "beta_2",
        "c_lambda",
        "crit_repeats",
        "device",
        "z_vec_demo",
    ],
)
