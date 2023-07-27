import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
oneByOne = False
plotBert = False
allCombinations = False

sub_lengths = [256, 320, 480]
codebooks_sizes = [750, 1874, 4124]
lrs_bert = [5e-5, 0.0001, 0.001]
lrs_mlp = [0.0001, 0.001, 0.01]
batch_sizes = [32, 64, 128]

sub_length = 256
codebook_size = 750
batch_size = 128
epochs = 50
lr = 0.0001
abspath = '/home/vinogradova/Documents/PycharmProjects/dn3-master/BENDRmain'
plotdir = f'{abspath}/BoW/results/plots/sleep-edf'

def plotMetric(data, network):
    # loss graph
    fig = plt.figure()
    plt.plot(data['Epoch'], data['Train_Loss'], label='Trainining')
    plt.plot(data['Epoch'], data['Val_Loss'], label='Validation')
    plt.legend()

    plt.title(f'Training loss - batch size {batch_size}')

    fig.savefig(f'{plotdir}/{network}_training_loss_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}.png')

    fig2 = plt.figure()

    plt.plot(data['Epoch'], data['Train_Accuracy'], label='Training')
    plt.plot(data['Epoch'], data['Val_Accuracy'], label='Validation')
    plt.legend()

    plt.title(f'Training accuracy - batch size {batch_size}')

    fig2.savefig(f'{plotdir}/{network}_training_accuracy_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}.png')

if not os.path.exists(plotdir):
    os.makedirs(plotdir)

if oneByOne:
    for sub_length in sub_lengths:
        for codebook_size in codebooks_sizes:
            #/home/vinogradova/Documents/PycharmProjects/dn3-master/BENDRmain/BoW/results/sleep-edf/mlp_s256_c750_batch128_e100_lr0.0001_train.csv
            file = f'{abspath}/BoW/results/sleep-edf/mlp_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}_split_train_corrCl.csv'
            data = pd.read_csv(file)
            plotMetric(data, 'mlp')

if plotBert:
    sub_length = 256
    codebook_size = 750
    batch_size = 80
    epochs = 40
    lr = 5e-05
    file = f'{abspath}/BoW/results/sleep-edf/bert_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}_split_train.csv'
    data = pd.read_csv(file)
    for i in range(len(data['Train_Accuracy'])):
        data['Train_Accuracy'][i] = float(data['Train_Accuracy'][i][7:13])
        data['Val_Accuracy'][i] = float(data['Val_Accuracy'][i][7:13])

    plotMetric(data, 'bert')


def plotAllCombinationsByTokenParams(metric):
    fig = plt.figure()
    for sub_length in sub_lengths:
        for codebook_size in codebooks_sizes:
            file = f'{abspath}/BoW/results/sleep-edf/mlp_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr0.0001_split_train_corrCl_best_model.csv'

            data = pd.read_csv(file)

            plt.plot(data['Epoch'], data[f'Val_{metric}'], label=f's{sub_length}_c{codebook_size}')
            plt.legend()
    plt.title(f'Validation {metric} - batch size {batch_size}')
    fig.savefig(f'{plotdir}/Validation_{metric}_batch{batch_size}_all.png')

def plotAllCombinationsByHyperparams(metric, mlpOrBert, trainVal, sub_length, codebook_size, epochs):
    fig = plt.figure()

    if mlpOrBert == 'bert':
        lrs = lrs_bert
    else:
        lrs = lrs_mlp

    for lr in lrs:
        for batch_size in batch_sizes:
            file = f'{abspath}/BoW/results/sleep-edf/{mlpOrBert}_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}_split_train.csv'

            data = pd.read_csv(file)

            #correct the mistake in recording where the accuracy was recorded as a tensor
            if type(data[f'{trainVal}_Accuracy'].values[0]) == str:
                for d in range(len(data)):
                    data.loc[d, f'{trainVal}_Accuracy'] = float(data.loc[d]['Val_Accuracy'][7:13])

            plt.plot(data['Epoch'], data[f'{trainVal}_{metric}'].values, label=f'lr={lr} | batch_size={batch_size}')
            plt.legend()
    plt.title(f'{trainVal} {metric}')
    fig.savefig(f'{plotdir}/{mlpOrBert}_{trainVal}_{metric}_by_hyperparams_all.png')



def makeResultsTableByHyperparams(metric, mlpOrBert, sub_length, codebook_size, epochs):
    if mlpOrBert == 'mlp':
        lrs = lrs_mlp
        results = {'0.0001': {'32': 0, '64': 0, '128': 0}, '0.001': {'32': 0, '64': 0, '128': 0}, '0.01': {'32': 0, '64': 0, '128': 0}}
    else:
        lrs = lrs_bert
        results = {'5e-05': {'32': 0, '64': 0, '128': 0}, '0.0001': {'32': 0, '64': 0, '128': 0}, '0.001': {'32': 0, '64': 0, '128': 0}}

    for lr in lrs:
        for batch_size in batch_sizes:
            file = f'{abspath}/BoW/results/sleep-edf/{mlpOrBert}_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}_split_test_corrCl_best_model.csv'
            test_results = pd.read_csv(file)
            if metric == 'Accuracy':
                results[str(lr)][str(batch_size)] = test_results[metric].values[0] * 100


    results_table = pd.DataFrame(results)
    results_table.to_csv(os.path.join(abspath, 'BoW', 'results', f'{mlpOrBert}_sleep-edf_results_{metric}_batch_all_e{epochs}_lr_all_best_model.csv'))



def makeResultsTableByTokenParams(metric):
    results = {'256': {'750': 0, '1874': 0, '4124': 0}, '320': {'750': 0, '1874': 0, '4124': 0}, '480': {'750': 0, '1874': 0, '4124': 0}}

    for sub_length in sub_lengths:
            file = f'{abspath}/BoW/results/sleep-edf/mlp_s{sub_length}_c{codebook_size}_batch{batch_size}_e50_lr0.0001_split_test_corrCl_best_model.csv'
            test_results = pd.read_csv(file)
            results[str(sub_length)][str(codebook_size)] = test_results[metric].values[0]

    results_table = pd.DataFrame(results)
    results_table.to_csv(os.path.join(abspath, 'BoW', 'results', f'sleep-edf_results_{metric}_mlp_batch{batch_size}_e{epochs}_lr0,0001_best_model.csv'))

def makeResultsTableConfusion(metric):
    metric_table = pd.DataFrame(columns=['Token_length', 'Codebook_size', 'Mov', 'SS_1', 'SS_2', 'SS_3', 'SS_4', 'Unscorable', 'REM', 'Wake'])
    for sub_length in sub_lengths:
        for codebook_size in codebooks_sizes:
            file = f'{abspath}/BoW/results/sleep-edf/mlp_s{sub_length}_c{codebook_size}_batch128_e50_lr0.0001_split_test_corrCl_best_model_conf.pkl'
            with open(file, 'rb+') as f:
                results = pickle.load(f)
            metric_data = results[metric]
            tmp = pd.DataFrame(columns=['Token_length', 'Codebook_size', 'Mov', 'SS_1', 'SS_2', 'SS_3', 'SS_4', 'Unscorable', 'REM', 'Wake'])
            tmp.loc[0, 'Token_length'] = sub_length
            tmp.loc[0, 'Codebook_size'] = codebook_size
            for i in range(len(metric_data)):
                column = tmp.columns[2+i]
                tmp.loc[0, column] = metric_data[i]
            metric_table = pd.concat([metric_table, tmp])


    metric_table.to_csv(os.path.join(abspath, 'BoW', 'results', f'sleep-edf_results_{metric}_mlp_batch{batch_size}_e{epochs}_lr0,0001_best_model.csv'))


def saveConfusionMatricesByHyperparams(mlpOrBert, sub_length, codebook_size, epochs):

    if mlpOrBert == 'bert':
        lrs = lrs_bert
    else:
        lrs = lrs_mlp

    for lr in lrs:
        for batch_size in batch_sizes:

            file = f'{abspath}/BoW/results/sleep-edf/{mlpOrBert}_s{sub_length}_c{codebook_size}_batch{batch_size}_e{epochs}_lr{lr}_split_test_corrCl_best_model_conf.pkl'
            with open(file, 'rb+') as f:
                results = pickle.load(f)
            confusion_matrix = pd.DataFrame(results['Confusion_Matrix'])
            confusion_matrix.to_csv(f'{abspath}/BoW/results/sleep-edf/{mlpOrBert}_Confusion_Matrix_{batch_size}_{lr}.csv')

makeResultsTableByHyperparams('Accuracy', 'bert', 256, 1874, 40)
plotAllCombinationsByHyperparams('Loss', 'bert', 'Train', 256, 1874, 40)
plotAllCombinationsByHyperparams('Accuracy', 'bert', 'Train')