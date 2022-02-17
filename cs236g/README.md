### Overview
This directory contains files for the CS236 term paper. Specifically, I am spinning up a GAN to simulate NFL passing plays, using NFL tracking data from the 2018 NFL regular season.

Some notable links:
 - Writeup location: Overleaf (Stanford Account)
 - [Data Source](https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
 - [BasketballGAN Paper](https://arxiv.org/pdf/1909.07088.pdf)

### Directory
The files in this repo are structured as follows:
 - `src`: contains all relevant code, including:
 	- `etl.py`: contains ETL functions, as well as a final `fetch_data()` function that pulls and processes
 	  all data for you.
 	- `constants.py`: contains relevant processing and modeling constants, such as play dimension, acceptable play personnel,
 	  and feature scaling values.
 	- `model.py`: contains functions to train, evaluate, and visualize your GAN. They should be utilized in one of the files in `src/scripts`.
 	- `scripts`: contains run-scripts to fit a particular model. Currently has `train_baseline.py` only; just call `python3 train_baseline.py` to
 	  deploy a training run.

### Notes
Lastly, as my AWS credit situation is still unresolved, the `src/scripts/train_baseline.py` script only functions for a small/toy dataset over a limited number of epochs on local CPU. Additional evaluation and model saving criteria will be pushed once VM/GPU situation is resolved. 
