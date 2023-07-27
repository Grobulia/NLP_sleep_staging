import glob
import os.path

import pandas as pd
import torch
import numpy as np
import tqdm
import argparse
import sys
import objgraph
import mne.io as loader
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from mne import pick_types, read_annotations, set_log_level

import time
import utils
from dn3.data.dataset import *
from result_tracking import ThinkerwiseResultTracker
from misc import *

from dn3.configuratron import ExperimentConfig
from dn3.utils import DN3ConfigException
from dn3.data.dataset import Thinker
import pickle
import feather


desired_sfreq = 256

def _prepare_session(raw, tlen, decimate, desired_sfreq, desired_samples, picks, exclude_channels, rename_channels,
                     hpf, lpf, DatasetConfig):
    if hpf is not None or lpf is not None:
        raw = raw.filter(hpf, lpf)

    lowpass = raw.info.get('lowpass', None)
    raw_sfreq = raw.info['sfreq']
    new_sfreq = raw_sfreq / decimate if desired_sfreq is None else desired_sfreq

    # Don't allow violation of Nyquist criterion if sfreq is being changed
    if lowpass is not None and (new_sfreq < 2 * lowpass) and new_sfreq != raw_sfreq:
        print(f"Could not create raw for {raw.filenames[0]}. With lowpass filter {raw.info['lowpass']}, sampling frequency {raw.info['sfreq']} and new sfreq {new_sfreq}. This is going to have bad aliasing!")
        return [], tlen, [], []
    else:
        # Leverage decimation first to match desired sfreq (saves memory)
        if desired_sfreq is not None:
            while (raw_sfreq // (decimate + 1)) >= new_sfreq:
                decimate += 1

        # Pick types
        picks = pick_types(raw.info, **{t: t in picks for t in DatasetConfig._PICK_TYPES}) \
            if DatasetConfig._picks_as_types(picks) else list(range(len(raw.ch_names)))
    
    
        # Pick types
        picks = pick_types(raw.info, **{t: t in picks for t in DatasetConfig._PICK_TYPES}) \
             if DatasetConfig._picks_as_types(picks) else list(range(len(raw.ch_names)))
        ch_selection = ["*PZ*", "*FPZ*"]
        if not any([fnmatch(raw.ch_names[idx], "*FPZ*") for idx in range(len(raw.ch_names))]):
             ch_selection.append("*FZ*")
        picks = ([idx for idx in picks if True in [fnmatch(raw.ch_names[idx], pattern) for pattern in ch_selection]])

        tlen = desired_samples / new_sfreq if tlen is None else tlen

        return raw, tlen, picks, new_sfreq

def constructDatasetAsBoW(ds_name, ds, desired_sfreq, abspath_bow, startingFileNo=0, clear_folder=False):
    output_folder = os.path.join(abspath_bow, 'pre_BoW_raw', f'pre_BoW_representation_{ds_name}_feather')
    # Outforlder = 'BENDRmain / BoW / pre_BoW_raw'

    if clear_folder:
        if os.path.exists(output_folder):
            clearOutputFolder(output_folder)


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    all_participants = {}
    annotations = []
    mapping = ds.auto_mapping()
    #print(mapping)

    fileNo = startingFileNo
    for thinker_id in tqdm.tqdm(mapping, unit='person'):
        thinker = mapping[thinker_id]
        for sess in thinker:
            sess = Path(sess)
            print(f'Thinker {thinker_id}, sess: {sess}')

            sess_id = ds._get_session_name(sess)

            #load raw data
            raw_data = loader.read_raw_edf(sess)

            raw, tlen, picks, new_sfreq = _prepare_session(raw_data, ds.tlen, ds.decimate, ds._sfreq, ds._samples, ds.picks,
                             ds.exclude_channels, ds.rename_channels, ds.hpf, ds.lpf, ds)
            
            if len(raw) > 0:
               #load annotations
               if ds_name == 'sleep-edf':
                   patt = ds.annotation_format.format(subject=thinker_id, session=sess_id)
                   ann_path = [str(f) for f in sess.parent.glob(patt)]
                   ann = mne.read_annotations(ann_path[0])
               
                   raw.set_annotations(ann)

               # #bring to a certain sampling frequency
               if raw.info['sfreq'] != desired_sfreq:
                   raw = raw.resample(sfreq=desired_sfreq)

            
               #filter out unnecessary channels
               raw_data, times = raw[:]
               selected_channels = {0: [], 1: []}

               for pick in picks:
                    if fnmatch(raw.ch_names[pick], "*FPZ*") | fnmatch(raw.ch_names[pick], "*FZ*"): 
                        selected_channels[1].append(raw_data[pick])
                    elif fnmatch(raw.ch_names[pick], "*PZ*"):
                        selected_channels[0].append(raw_data[pick])
               
               #print(selected_channels)
               if len(selected_channels[0]) > len(selected_channels[1]):
                    selected_channels[1].append([])
               elif len(selected_channels[0]) < len(selected_channels[1]):
                    selected_channels[0].append([])
                   

               #print(picks)
               #print(selected_channels)
               #append channel-by-channel
               for pick in range(2):
                   channelName = f'ch{pick}'
                   
                   if channelName not in all_participants.keys():
                       all_participants[channelName] = {'data': [selected_channels[pick]], 'times': [times]}
                   else:
                       all_participants[channelName]['data'].append(selected_channels[pick])
                       all_participants[channelName]['times'].append(times)

               if ds_name == 'sleep-edf':
                   events = mne.events_from_annotations(raw, chunk_duration=30)
                   annotations.append(events)
               
               if (len(all_participants['ch0']['data']) == 100) | (thinker_id == list(mapping)[-1]):
                   print('Saving')
                   #print(all_participants)
                   for channel in all_participants:
                        #with open(f'{output_folder}/prebow_{fileNo}_{channel}.pkl', 'wb+') as f: #TODO: make sure there are no duplicate entries in the file
                        #    pickle.dump(all_participants, f)
                        #f.close()
                        #np.save(f'{output_folder}/prebow_{fileNo}_{channel}.npy', all_participants[channel])
                        all_participants_df = pd.DataFrame(data=all_participants[channel]).reset_index()
                        all_participants_df.to_feather(f'{output_folder}/prebow_{fileNo}_{channel}.feather')


                        print('Channels saved')
        	
                    
                   with open(f'{output_folder}/annotations_{fileNo}.pkl', 'wb+') as f: #TODO: make sure there are no duplicate entries in the file
                        pickle.dump(annotations, f)

                   print('Annotations saved')
        	
                   fileNo += 1
                   all_participants = {'data': [], 'times': []}
                   annotations = []

            


        





