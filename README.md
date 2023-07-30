# NLP_sleep_staging

## Introduction

This document contains the instructions on how to run the codebase for the Master’s Thesis *More than a thousand words: an application of BERT and Bag-of-Words to EEG data for Sleep Stage Classification*. The codebase can be found by following this link - [GitHub Repository](https://github.com/Grobulia/NLP_sleep_staging/).

The goal of the thesis was to use the BERT model on EEG data tokenised with the Bag-of-Words method (Wang et al., 2013) for the purpose of sleep-stage classification. A multilayer perceptron trained on the same data, as well as two networks from the literature - BENDR (Kostas et al., 2021) and CatBoost (Donckt et al., 2022) were used for comparison. One dataset was used for pretraining and one for the classification task. The repository has the following structure:

BENDR
|
├── BENDRMain                    
│   ├── BoW          
│   ├── configs         
│   └── dn3                


In this, some of the code was borrowed from the repository of the BENDR network [GitHub Repository](https://github.com/SPOClab-ca/BENDR) and the supporting EEG-processing library dn3 [GitHub Repository](https://github.com/SPOClab-ca/dn3). Minor local modifications have been made for the purpose of this study.

### Main Folder BENDR/BENDRmain

The main entry in the repository, containing the subfolders and files relating to BENDR, tokenisation, and BERT.

**General:**

- `dataset_descriptives.py`: Contains functions to run the descriptive statistics of the datasets in terms of the total recording time.
- `misc.py`: Miscellaneous functions for terminal input parsing, determining the number of files in the dataset, etc.

**BENDR:**

- `analysis.py`: Functions relating to the calculation of evaluation metrics and results visualization.
- `dn3_ext.py`: Supporting functions and classes relating to the execution of the BENDR network.
- `downstream.py`: Script for the fine-tuning of the pretrained BENDR network. Dependencies: pretrained models in the BENDRMain\checkpoints folder.
- `pretrain.py`: Script for the pretraining of the BENDR network. Dependencies: raw data in the BENDRMain\datasets folder.
- `sequence_prediction.py`: Script for the evaluation of the Masked Language Modelling task performed during pretraining. Dependencies: pretrained models in the BENDRMain\checkpoints folder.
- `utils.py`: Miscellaneous utility functions.
- `results_tracking.py`: Functions for organizing the results during training.

**configs (folder):**

A folder containing the config files for the pretraining, fine-tuning, and sequence evaluation. The config files specify which datasets are used in the stage and the dataset characteristics, such as which channels are included or excluded, the required sampling frequency, and which metrics are used for evaluation of downstream tasks.

**dn3 (folder):**

Library for EEG processing and analysis. See - [GitHub Repository](https://github.com/SPOClab-ca/dn3) for more information. Required for all of the above.

### Bag-of-Words tokenisation

Files for server execution:

- `run_bow_construct_dataset.sh`, `run_bow_representation.sh`, `run_bow_codebook.sh`, `run_bow_vq.sh`: Shell scripts to execute respective stages of tokenisation on the server.
- `do_all_construct_dataset.py`, `do_all_representation.py`, `do_all_codebook.py`, `do_all_vq.py`: Python scripts to execute respective stages of tokenisation with specified parameters.

**BoW (folder):**

- `analysis.py`: Contains functions for the visualization of the results in plot and table form. Dependencies: results of the classification for BERT or MLP in the BoW\results folder.
- `BoW_data_loader.py`: Stage 1 of tokenisation. The script to preprocess the raw data by bringing it to the same sampling rate and number of channels. Saves the resulting data in sets of 5 (Physionet Sleep-EDF) or 100 (TUEG) recordings in the feather format. Dependencies: raw data in the BENDRMain\datasets folder.
- `do_representation.py`: Stage 2 of tokenisation. The script to perform the segmentation of the data by the sliding window and apply the wavelet transform to each segment. Saves each recording as a set of the resulting vectors, preserving the structure from stage 1. Dependencies: output of stage 1 in the BoW\pre_BoW_raw folder.
- `do_codebook.py`: Stage 3 of tokenisation. The script to perform the k-means clustering algorithm on the vectors from stage 2. Saves the resulting cluster centres. Dependencies: output of stage 2 in the BoW\feature folder.
- `do_vq.py`: Final stage of tokenisation. The script to convert each recording into a sequence of vector centroids. Saves the recordings in the same file structure as stage 1. Dependencies: output of stage 2 in the BoW\feature folder and stage 3 in the BoW\codebook folder.
- `MLP_new_server.py`: Perform MLP classification on the tokenised recordings. Dependencies: output of stage 2 in the BoW\closest_centres folder and annotations from the BoW\pre_BoW_raw folder.

### BERT

- `prepareInputForBERT.py`: Script to prepare the tokenised data for the transformer training. Dependencies: output of stage 2 in the BoW\closest_centres folder and (for fine-tuning) annotations from the BoW\pre_BoW_raw folder.
- `BERT.py`: The script for BERT pretraining on the tokenised data. Dependencies: inputs data from prepareInputForBERT.py.
- `BERT_fine_tuning.py`: The script for the fine-tuning of the pretrained BERT model. Dependencies: pretrained model in the root folder and ...
- `BERT_fine_tuning_test.py`: The script to test the fine-tuned BERT model.

### Execution sequence for the BENDR model:

- `Pretrain.py`
- `Downstream.py` for each desired configuration.

### Execution sequence for the tokenisation:

- `BoW_data_loader.py`
- `Do_representation.py` through `do_representation.py`
- `Do_codebook.py` through `do_codebook.py`
- `Do_vq.py` through `do_vq.py`
- `MLP_new_server.py`

### Execution sequence for BERT:

- BoW tokenisation for both datasets
- `BERT.py`
- `BERT_fine_tuning.py`
- `BERT_fine_tuning_test.py`

**Requirements:**

- feather==0.1.2
- feather_format==0.4.1
- h5py==3.1.0
- kmeans_pytorch==0.3
- matplotlib==3.3.4
- memory_profiler==0.61.0
- mne==0.23.4
- moabb==0.5.0
- numpy==1.19.5
- objgraph==3.5.0
- opencv_python==4.7.0.72
- pandas==1.1.1
- parse==1.15.0
- pyclustering==0.10.1.2
- PyWavelets==1.1.1
- PyYAML==6.0.1
- pyyaml_include==1.2
- scikit_learn==1.3.0
- scipy==1.5.4
- seaborn==0.11.2
- torch==1.9.0
- torchmetrics==0.8.2
- torchtext==0.10.0
- tqdm==4.64.1
- transformers==4.18.0
