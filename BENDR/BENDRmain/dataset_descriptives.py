import os.path

import pandas as pd
import torch
import tqdm
import argparse
import sys
import objgraph
import mne.io as loader
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from mne import pick_types, read_annotations, set_log_level
from BoW.BoW_data_loader import _prepare_session

import time
import utils
from dn3.data.dataset import *
from result_tracking import ThinkerwiseResultTracker

from dn3.configuratron import ExperimentConfig

datasetConfigs = ['pretraining', 'downstream']

experiment = ExperimentConfig(f'./configs/{datasetConfigs[0]}.yml')

for ds_name, ds in tqdm.tqdm(experiment.datasets.items(), total=len(experiment.datasets.items()), desc='Datasets'):
    mapping = ds.auto_mapping()
    total_time = 0
    for t in tqdm.tqdm(mapping, unit='person'):
        thinker = mapping[t]
        for sess in thinker:
            sess = Path(sess)
            print(sess)
            raw_data = loader.read_raw_edf(sess)
            total_time += raw_data.times[-1]

    print(f'\n\n\n{ds_name} - total time: {round(total_time)}')
