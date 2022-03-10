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
 	- `train.py`: contains the script to train the model. Choose your hyperparams in `CONFIGS` at the type of the file (eventually this will be done via `.json` or `argparse`, but for now configuration is manual). Then run the script; results save to drive.

### Notes
While this repo can ostensibly be run anywhere, my recommendation would be to open it through Colab, and fit there. For one, Colab's GPU is pretty functional;
moreover, it makes it incredibly easy to save your training results to drive, without much `.pkl` or other file hassle. Note here that before running, make sure your drive and/or filepath has the following folders:
- `./CS236G`
- `./CS236G/data/`
- `./CS236G/runs/`

### Requirements
Run `pip install -r requirements.txt` or `pip install -r requirements_vanilla.txt` before training. Note that `requirements.txt` is guaranteed to work, but `requirements_vanilla.txt` may be more compatible with wherever the code is being run. 
