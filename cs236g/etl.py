"""Ingest and processing helpers"""

### general imports
import os
import pickle
import gc
import random
import itertools
import datetime as dt
import numpy as np
import pandas as pd

# from scipy.stats import norm
from tqdm import tqdm, trange

### viz imports
import matplotlib.pyplot as plt
import seaborn as sns

### GCloud imports
from google.colab import drive

drive.mount("/content/drive")

### utils imports
from constants import *

### configs
pd.set_option("chained_assignment", None)


def save_pickle(file, filename):
    """
    Pickles something
    """
    with open(filename, "wb") as handle:
        pickle.dump(file, handle)


def load_pickle(filename):
    """
    Reads back in a pickled object
    """
    with open(filename, "rb") as handle:
        result = pickle.load(handle)
    return result


def merge_plays_with_tracking(
    tracking_df, plays_df, personnel_types=OFF_PERSONNEL_TYPES
):
    """
    Merges the play df onto the tracking df, and filters down to relevent personnel types
    Args:
      tracking_df : pd.DataFrame
        The third output of fetch_data()
      plays_df : pd.DataFrame
        The second output of fetch_data()
      personnel_types : list[str]
        Personnel types -- in the format of plays_df.personnelO -- to be included
        in the ultimate dataset. Recommend using OFF_PERSONNEL_TYPES
    Returns : pd.DataFrame
      A dataframe containing play and tracking data
    """
    ### read plays, and restrict to personnel types of interest
    plays_df = plays_df.loc[(plays_df.personnelO).isin(personnel_types), :].reset_index(
        drop=True
    )
    ### join the play information, and restrict to regulation
    ### also ensure
    plays_tracking = (
        pd.merge(plays_df, tracking_df, on=["gameId", "playId"])
        .query("quarter <= 4")
        .dropna(subset=["absoluteYardlineNumber"])
    )

    ### have to parse the play direction
    # is the offensive team on their side of the field
    plays_tracking["same_side_flag"] = [
        1 if x == y else 0
        for (x, y) in list(
            zip(
                plays_tracking.possessionTeam.values, plays_tracking.yardlineSide.values
            )
        )
    ]
    ### split 50 yard line plays -- harder to extract direction here
    plays_tracking_los_50 = plays_tracking.query("yardlineNumber == 50")
    ### for now, omit 50 yard line -- unable to extract direction
    plays_tracking = plays_tracking.query("yardlineNumber != 50")

    ### Rules for flip ###
    # same_side_flag == 1 & absoluteYardlineNumber < 60 --> up (1)
    # same_side_flag == 0 & absoluteYardlineNumber > 60 --> up (1)
    # same_side_flag == 1 & absoluteYardlineNumber > 60 --> down (-1)
    # same_side_flag == 0 & absoluteYardlineNumber < 60 --> down (-1)

    ### rearrange all plays to have vertical orientation ###
    plays_tracking["vertical_orientation"] = [
        1 if ((x == 1 and y < 60) or (x == 0 and y > 60)) else 0
        for (x, y) in list(
            zip(
                plays_tracking.same_side_flag.values,
                plays_tracking.absoluteYardlineNumber.values,
            )
        )
    ]

    ### apply processed coordinate // flip if appropriate
    ## note dim change to make plays vertical and upwards

    ### raw x/y coordinates, for debugging etc.
    # y
    plays_tracking["y_coord"] = plays_tracking["x"].values * plays_tracking[
        "vertical_orientation"
    ].values + (120 - plays_tracking["x"].values) * (
        1 - plays_tracking["vertical_orientation"].values
    )
    # x
    plays_tracking["x_coord"] = plays_tracking["y"].values * plays_tracking[
        "vertical_orientation"
    ].values + (
        160 / 3 - plays_tracking["y"].values
    ) * (  # field is 160 yards wide
        1 - plays_tracking["vertical_orientation"].values
    )

    ### line of scrimmage in standardized coords
    plays_tracking["line_of_scrimmage_coord"] = plays_tracking[
        "absoluteYardlineNumber"
    ].values * plays_tracking["vertical_orientation"].values + (
        120 - plays_tracking["absoluteYardlineNumber"].values
    ) * (
        1 - plays_tracking["vertical_orientation"].values
    )

    ### each player's offset relative to LoS
    plays_tracking["y_coord_los"] = (
        plays_tracking["y_coord"].values
        - plays_tracking["line_of_scrimmage_coord"].values
    )
    return plays_tracking


def test_play_statistic_timestamps(play_statistic_full):
    """
    Checks to make sure timestamps are consistent for all players on a given
    play.
    """
    # look at number of timestamps per play
    play_pos_counts = play_statistic_full.copy()
    play_pos_counts["n"] = 1.0
    play_pos_counts = play_pos_counts.groupby(
        ["position", "nflId", "gameId", "playId"], as_index=False
    )["n"].sum()

    # make sure everyone stops and starts at same time
    play_minmax = pd.merge(
        play_pos_counts.groupby(
            ["position", "nflId", "gameId", "playId"], as_index=False
        )["n"].min(),
        play_pos_counts.groupby(
            ["position", "nflId", "gameId", "playId"], as_index=False
        )["n"].max(),
        on=["position", "nflId", "gameId", "playId"],
        suffixes=["_min", "_max"],
    )
    # check for mismatched plays, by number of timestamps
    mismatched_plays_n = play_minmax.query("n_min != n_max")
    assert mismatched_plays_n.shape[0] == 0


def test_dupe(df):
    """
    On a handful of occasions, the raw NFL ops feed will provide duped
    tracking coords, e.g. every frame is just added twice. Relevant examples include:
    * gameId = 2018123001 // playId == 435
    * gameId = 2018091605 // playId == 2715
    On inspection, the tracking coords -- while duped -- seem fine. But because they pop up
    so infrequently, I'm reluctant to proceed with them unless I fully knew the issue at hand.
    Hence, this function checks for such dupes.
    Args:
      df : pd.DataFrame
        A pandas df of tracking data for a single play
    Returns : bool
      Whether or not there is a dupe (True) or not (False)
    """
    # should only be one measurement per timestamp
    assert (
        df.groupby(
            ["gameId", "playId", "nflId", "time_str", "frameId"], as_index=False
        )["x"]
        .count()
        .x.max()
        == 1
    )


def test_skill_personnel_count(df):
    """
    Function that tests whether six skill players are on the field for the start of the play,
    per football rules
    Args:
      df : pd.DataFrame
        A pandas df of tracking data for a single play
    """
    ### get first measurements
    initial_positions = (
        df.sort_values(["nflId", "time"], ascending=True)
        .groupby(["nflId"], as_index=False)
        .head(1)
    )
    ### check against eligibility: need 6 skill guys on the field at a time
    assert initial_positions.shape[0] == 6


def test_play_length(play_snapshot):
    """
    Makes sure data has sufficiently many rows, and is not a spike.
    Args:
      play_snapshot : pd.DataFrame
        A pandas df, as constructed in parse_passing_plays().
    """
    assert play_snapshot.play_frames.iloc[0] > 0


def test_play_integrity(tracking_df, play_snapshot):
    """
    Applies check_dupe(), check_personnel_count(), and ensures play is not a spike.
    Args:
      tracking_df : pd.DataFrame
        A pandas df of tracking data for a single play
      play_snapshot : pd.DataFrame
        A pandas df, as constructed in parse_passing_plays()
    """
    ### ensure rows exist
    test_play_length(play_snapshot)
    ### ensure not a dupe
    test_dupe(tracking_df)
    ### ensure personnel
    test_skill_personnel_count(tracking_df)


def plot_play(df, figsize=(12, 6)):
    """
    Visualizes the routes associated with a particular play,
    for debugging and/or general understanding
    Args:
      df : pd.DataFrame
        A dataframe, as wrangled + outputted by merge_plays_with_tracking()
    """
    ### figure size
    plt.rcParams["figure.figsize"] = figsize
    ### plot line of scrimage
    plt.plot(
        [0, 160 / 3],
        [
            df.line_of_scrimmage_coord_raw.iloc[0],
            df.line_of_scrimmage_coord_raw.iloc[0],
        ],
        label="LoS",
        color="black",
    )
    plt.plot(
        [0, 160 / 3],
        [
            df.line_of_scrimmage_coord_raw.iloc[0] + 10,
            df.line_of_scrimmage_coord_raw.iloc[0] + 10,
        ],
        label="LoS + 10",
        color="black",
        linestyle="dashed",
    )
    ### plot everybody's routes
    for name in df.displayName.unique():
        df_route = df.query(f"displayName == '{name}'")
        plt.plot(df_route.x_coord_raw.values, df_route.y_coord_raw.values, label=name)
    ### extract play descriptors
    yardline_side = df_route.yardlineSide.values[0]
    yardline_num = df_route.yardlineNumber.values[0]
    abs_yardline_num = df_route.absoluteYardlineNumber.values[0]
    poss_team = df_route.possessionTeam.values[0]
    quarter = df_route.quarter.values[0]
    play_dir = df_route.playDirection.values[0]
    ### go back to plotting
    plt.legend()
    plt.title(
        f"""
        Yard Side: {yardline_side} || Yard #: {yardline_num} || Abs(Yard #): {abs_yardline_num}
        Poss. Team: {poss_team} || Qtr: {quarter} || Play Dir: {play_dir}
        """
    )
    plt.show()


def parse_passing_plays(plays_tracking, verbose=False):
    """
    Given a dataframe outputted by merge_plays_tracking(), this function
    iterates through each play, extracts the relevant route/play window, and then performs
    basic integrity tests.
    Args:
      plays_tracking : pd.DataFrame
        A dataframe outputted by merge_plays_tracking()
      verbose : bool
        Whether or not to plot the plays as you parse them
    Returns : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
      * A dataframe of parsed/tested tracking data
      * A dataframe of play dimensions
      * A dataframe of plays that failed the data integrity checks, for manual inspection
    """
    # if you're plotting, don't do TQDM
    if verbose:
        vfun = lambda x: x
    else:
        vfun = tqdm

    ### this contains the tracking data for each play
    play_statistic_full = []
    ### this contains the dims/sizes of each play
    play_frame_dims = []
    ### this contains plays that failed to process, for inspection
    failed_plays = []

    for grp, df in vfun(
        plays_tracking.sort_values(["gameId", "playId", "time_str"]).groupby(
            ["gameId", "playId"]
        )
    ):

        try:
            ### start of play, for model purposes
            start_time = (
                df.query("event == 'ball_snap'").sort_values("time_str").head(1)
            )
            ### end of play, for model purposes
            end_time = (
                df.query(
                    "event in ['pass_arrived', 'qb_sack', 'pass_outcome_incomplete', 'interception', 'qb_strip_sack']"
                )
                .sort_values("time_str")
                .head(1)
            )

            ### filter tracking to range
            df_play_window = df.query(
                f"time_str < '{end_time.time_str.values[0]}' & time_str >= '{start_time.time_str.values[0]}' "
            )

            ### compute play dims for data integrity
            play_snapshot = pd.DataFrame(
                {
                    "gameId": df_play_window.gameId.values[0],
                    "playId": df_play_window.playId.values[0],
                    "play_frames": [df_play_window.query("position == 'QB'").shape[0]],
                    "play_time": [
                        (
                            df_play_window.time.max() - df_play_window.time.min()
                        ).total_seconds()
                    ],
                }
            )

            ### does it fail due to a known data quality issue?
            try:
                test_play_integrity(df_play_window, play_snapshot)
                ### if the play passes muster, add it to the heap
                play_frame_dims.append(play_snapshot)
                play_statistic_full.append(df_play_window)
            except AssertionError:
                df_play_window["reason"] = "AssertionError"
                failed_plays.append(df_play_window)

            if verbose:
                pass
        ### otherwise it failed to an unknown issue
        except:
            df["reason"] = "other"
            failed_plays.append(df)
    gc.collect()
    ### put everything together
    play_statistic_full = pd.concat(play_statistic_full).reset_index(drop=True)
    play_frame_dims = pd.concat(play_frame_dims).reset_index(drop=True)
    failed_plays = pd.concat(failed_plays).reset_index(drop=True)
    gc.collect()
    return (play_statistic_full, play_frame_dims, failed_plays)


def make_route_tensor_player(
    player_df, n_frames=N_FRAMES, center_x=False, center_y=False
):
    """
    Converts a df of a player's positioning within a particular passing play
    into a tensor for the purposes of modeling. Extracts coordinates, speed, and
    acceleration, stacks these into a matrix, and then pads with zeros as necessary
    so that the matrix is of a standard size. Notably, the "is_route_over" indicator
    serves as a flag that the route-running is over, and that the coordinates/speed/
    acceleration in the matrix are indeed padded.
    Args:
      player_df : pd.DataFrame
        A df for a single player/play pairing with `time`, `x_coords`, `y_coords`, `s` and `a`
        fields
      n_frames : int
        Number of frames for return matrix
      center_x : bool
        Whether or not to center the route on the x axis
      center_y : bool
        Whether or not to center the route on the y-axis
    Returns : np.array(N_frames, 5)
      A tensor of positioning data for a player and play.
    """

    ### make a tensor of player
    player_df_tidy = player_df.sort_values("time")[PLAYER_TENSOR_COLS].head(n_frames)
    player_tensor = player_df_tidy.values  # (N_FRAMES x FRAME_WIDTH)
    if center_y:
        ### center LoS (y-coord)
        player_tensor[:, 1] = player_tensor[:, 1] - player_tensor[0, 1]
    if center_x:
        ### center LoS (y-coord)
        player_tensor[:, 0] = player_tensor[:, 0] - player_tensor[0, 0]
    return player_tensor


def make_route_tensor_play(df, n_frames=N_FRAMES):
    """
    Makes a tensor describing an entire route sequence, of the following form:
    * for each of the 5 skill positions, the player's X, Y, S, A are horizontally
      concatenated.
    * Order of this concatenation is determined by starting position, as dictated
      by L->R (ascending x-coord) ordering at the time of ball snap. That is, the leftmost
      non-QB skill player on the field corresponds to the first four columns of the tensor, the
      second leftmost non-QB skill player on the field corresponds to the second four columns of the tensor,
      etc.
    * The QB is appended last (four columns, in similar fashion), after the five skill players
    """

    ###
    initial_positions = (
        df.sort_values(["nflId", "time"], ascending=True)
        .groupby(["nflId"], as_index=False)
        .head(1)
    )
    ### check against eligibility: need 6 skill guys on the field at a time
    assert initial_positions.shape[0] == 6

    ### extract QB; he's position last in tensor
    qb_id = initial_positions.query("position == 'QB'").nflId.values[0]
    ### then position everybody L-R
    skill_init_pos = initial_positions.query("position != 'QB'").sort_values("x_coord")
    ### init list of route tensors
    route_tensors_all = []

    ### iterate over skill positions and make tensors
    for nfl_id in list(skill_init_pos.nflId.values) + [qb_id]:
        player_df = df.query(f"nflId == {nfl_id}").sort_values("time")
        route_tensors_all.append(make_route_tensor_player(player_df, n_frames=n_frames))
    ### combine route tensors: again, L->R, then QB
    route_tensors_all = np.hstack(route_tensors_all)
    ### add on an indicator to show that play is not yet over: -1 if alive, 1 if done
    route_tensors_all = np.hstack(
        [route_tensors_all, -np.ones((route_tensors_all.shape[0], 1))]
    )
    ### pad at bottom if play completed before n frames
    if route_tensors_all.shape[0] < n_frames:
        n_pad = n_frames - route_tensors_all.shape[0]
        ### switch on the indicator that play is over
        pad_tensor = np.hstack([np.zeros(route_tensors_all.shape[1] - 1), 1.0])
        # use final Y's
        pad_tensor[np.arange(1, 25, 4)] = route_tensors_all[-1, np.arange(1, 25, 4)]
        # use final X's
        pad_tensor[np.arange(0, 25, 4)] = route_tensors_all[-1, np.arange(0, 25, 4)]
        pad_tensor = np.vstack([pad_tensor for _ in range(n_pad)])
        route_tensors_all = np.vstack([route_tensors_all, pad_tensor])
    # gc.collect()
    return route_tensors_all


def make_route_tensors(play_statistic_full):
    """
    Makes route tensors, for modeling purposes.
    Args:
      play_statistic_full : pd.DataFrame
        A dataframe, as returned in the first index by parse_passing_plays()
    """
    failures, routes, corresponding_data = [], [], []
    for grp, df in tqdm(
        play_statistic_full.groupby(["gameId", "playId"], as_index=False)
    ):
        try:
            route = make_route_tensor_play(df=df, n_frames=N_FRAMES)
            routes.append(route)
            corresponding_data.append(df)
        except:
            failures.append(df)
    routes = np.stack(routes)
    gc.collect()
    return (
        np.swapaxes(routes, 1, 2),  # flip to (N, FRAME_WIDTH, N_FRAMES)
        corresponding_data,
        failures,
    )


def fetch_data_byweek(filepath=FILEPATH + "big-data-bowl-2021/", week_num=2, **kwargs):
    """
    Pulls game data, play data, and tracking data from Drive for a single
    week of play
    Args:
      filepath : str
        Filepath prefix from which to pull data. Should have files:
        * games.csv
        * plays.csv
        * week<week_start>.csv
        * ...
        * week<week_end>.csv
      week_num : int
        Week number (between 1 and 17) to fetch tracking data from
    Returns : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
      Dataframes containing:
      * game information
      * play information
      * tracking information
    """
    print(f"...Reading in Week {week_num} data...")
    ### read games (all games)
    # games = pd.read_csv(filepath + "games.csv")
    ### read plays (all weeks)
    plays = pd.read_csv(filepath + "plays.csv")
    ### read tracking within particular week
    tracking = pd.read_csv(filepath + f"week{week_num}.csv").query(
        "position in ('QB', 'WR', 'RB', 'TE', 'HB', 'FB')"
    )
    ### handle times
    tracking["time_str"] = [str(item) for item in tracking.time]
    tracking["time"] = [pd.Timestamp(item) for item in tracking.time]
    plays_tracking = merge_plays_with_tracking(tracking, plays, **kwargs)

    (play_statistic_full, play_frame_dims, failed_plays) = parse_passing_plays(
        plays_tracking
    )
    gc.collect()
    ### lets do those tensors
    return make_route_tensors(play_statistic_full)


def fetch_data(week_start, week_end):
    """
    Gets all data relevant to the project.
    Args:
      week_start : int
        Starting week to pull data from, in [1, week_end]
      week_end : int
        Ending week to pull data from, in [week_start, 17]
    Returns : tuple[np.array, list[pd.DataFrame], list[pd.DataFrame]]
      * the route tensors, for use in modeling
      * a list of dataframes, each with tracking data corresponding to the first
        axis of the route tensors
      * a list of plays that failed to parse, for manual inspection.
    """
    routes, play_info, failures = [], [], []
    ### iterate over weeks and parse passing data
    ### doing it this way saves memory
    for wk in range(week_start, week_end + 1):
        routes_temp, play_info_temp, failures_temp = fetch_data_byweek(week_num=wk)
        routes.append(routes_temp)
        play_info += play_info_temp
        failures += failures_temp
    routes = np.concatenate(routes, axis=0)
    gc.collect()
    return routes, play_info, failures


def make_play_z_inits(play_info_df):
    """
    Extracts the starting positions, and one-hot route labels, for all skill players
    in the route.

    Args:
      play_info_df : pd.DataFrame
        A dataframe -- likely to be an element of play_info, the second output of fetch_data
    Returns : np.array
      A vector containing starting position info
    """
    # filter down to first frame
    play_info_df_ = play_info_df.query(f"time_str == '{min(play_info_df.time_str)}'")
    ### use this to organize s.t. QB goes last
    play_info_df_["pos_sort_idx"] = [
        1 if item != "QB" else 2 for item in play_info_df_.position
    ]
    play_info_df_ = play_info_df_.sort_values(
        ["pos_sort_idx", "x_coord"], ascending=True
    )
    ### make route indicators
    for route_name in ROUTE_LABELS:
        play_info_df_[f"is_route_{route_name}"] = (
            play_info_df_["route"].values == route_name
        ).astype(int)
    ### collapse down as a vector
    # initial positions first
    z_supp_pos = play_info_df_[["x_coord", "y_coord_los", "s", "a"]].values.ravel()
    # route tags
    z_supp_route = play_info_df_[
        [f"is_route_{route_name}" for route_name in ROUTE_LABELS]
    ].values.ravel()
    # QB doesn't get a route, so remove
    z_supp_route = z_supp_route[: -len(ROUTE_LABELS)]
    return np.hstack([z_supp_pos, z_supp_route])


def center_routes(routes):
    """
    Centers each route such that it begins from the (0, 0) coordinate
    """
    routes = routes.copy()
    routes[:, np.arange(0, 24, 4), :] = np.subtract(
        routes[:, np.arange(0, 24, 4), :], routes[:, np.arange(0, 24, 4), 0, None]
    )
    routes[:, np.arange(1, 24, 4), :] = np.subtract(
        routes[:, np.arange(1, 24, 4), :], routes[:, np.arange(1, 24, 4), 0, None]
    )
    return routes


def scale_routes(routes):
    """
    Custom to each variable, coerces down to (-1, 1) scale
    """
    routes = routes.copy()
    ### scale X
    routes[:, np.arange(0, 24, 4), :] = (routes[:, np.arange(0, 24, 4), :] + 45) / (
        45 + 45
    )
    ### scale Y
    routes[:, np.arange(1, 24, 4), :] = (routes[:, np.arange(1, 24, 4), :] + 15) / (
        55 + 15
    )
    ### speed
    routes[:, np.arange(2, 24, 4), :] = (routes[:, np.arange(2, 24, 4), :] + 0) / (
        12 - 0
    )
    ### accel
    routes[:, np.arange(3, 24, 4), :] = (routes[:, np.arange(3, 24, 4), :] + 0) / (
        12 - 0
    )

    ### (-1 ,1)
    routes = routes * 2 - 1
    routes[routes < -1.0] = -1.0
    routes[routes > 1.0] = 1
    return routes
