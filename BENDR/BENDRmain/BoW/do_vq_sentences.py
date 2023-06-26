import os
import pandas
import pandas as pd
import numpy as np
import pywt as pywt
#from BoW.configs import *
from BENDRmain.misc import *
import pickle
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def do_vq(runNo, abspath, dataSource, codebook_size, sub_length, inter_point, max_iterations, startFileNo, chNo, dataset='test', makeHistogram=False):
    output_folder = os.path.join(abspath, 'closest_centres', dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(os.path.join(abspath, 'codebook', dataset, f'kmean_s{sub_length}_i{inter_point}_c{codebook_size}.pkl'), 'rb') as f:
        centres = pickle.load(f)
    print(centres)
    if not os.path.exists(f'results/{dataset}'):
        os.makedirs(f'results/{dataset}')
    
    noChannels = determine_no_channels_from_dataset(dataset)
    noFiles = determine_no_of_data_files_in_dataset(dataset, abspath)

    
    if dataSource == 'csv':
        for i in range(1, 6): #loop through conditions

            print(f'VQ Condition {i}')
            with open(os.path.join(abspath, f'feature/{dataset}/data_tran_{i}.pkl'), 'rb') as f:
                data_tran = pickle.load(f)

            output_file = os.path.join(output_folder, f'closest_centres_{i}.pkl')

            # the function will also save descriptors in files
            histogram = makeDescriptorVQ(data_tran, codebook_size, centres, output_file, makeHistogram)

            if makeHistogram:
                plotHistogram(histogram, dataset, runNo, i)

    elif dataSource == 'pickle':
        channelNo = chNo
        print(f'VQ Channel {channelNo}')
            
        for fileNo in range(startFileNo, noFiles):
                with open(os.path.join(abspath, f'feature/{dataset}/data_tran_s{sub_length}_i{inter_point}_ch{channelNo}_{fileNo}.pkl'), 'rb') as f:
                    data_tran = pickle.load(f)

                if makeHistogram:
                    histogram = [0] * codebook_size

                output_file = os.path.join(output_folder, f'closest_centres_s{sub_length}_i{inter_point}_c{codebook_size}_ch{channelNo}_{fileNo}_sent.pkl')
                #the function will also save descriptors in files
                histogram = makeDescriptorVQ(data_tran, sub_length, codebook_size, centres, output_file, fileNo, makeHistogram)

                if makeHistogram:
                    plotHistogram(histogram, dataset, runNo, i)




def makeDescriptorVQ(data_tran, sub_length, codebook_size, centres, output_file, fileNo, makeHistogram=False):
    histogram = []
    if makeHistogram:
        histogram = np.zeros(codebook_size)
    descriptor_vq = []
    
    for dataSampleIndex in data_tran:
        dataSampleWhole=data_tran[dataSampleIndex]
        
        dataSampleWhole = np.transpose(dataSampleWhole)
        print(len(dataSampleWhole))


        # find number of points per timeseries
        nPoints = len(dataSampleWhole)

        tokenIndices = list(range(0, nPoints, sub_length // 4))
        dataSample = dataSampleWhole[tokenIndices]
        dataSampleWhole=[]
        print(dataSample.shape)

        print(np.array(centres).shape)
     
        distanceMat = cdist(dataSample, centres)
        dataSample = []

        descriptor_vq.append(np.argmin(distanceMat, axis=1))
        distanceMat = []

    with open(output_file, 'wb+') as f:
            pickle.dump(descriptor_vq, f)
    
    descritor_vq = []
    return histogram

def plotHistogram(histogram, dataset, runNo, i):
    unique, counts = np.unique(histogram, return_counts=True)
    counts2, bins = np.histogram(histogram, bins=unique)
    fig = plt.figure()
    plt.hist(histogram, bins=bins)
    edgeTicks = list(range(max(bins) + 1))
    # for tick in edgeTicks:
    #     if tick % 100 != 0:
    #         edgeTicks[edgeTicks.index(tick)] = ''
    plt.xticks(edgeTicks)
    fig.savefig(f'results/{dataset}/histogram_{runNo}_{i}.png')
