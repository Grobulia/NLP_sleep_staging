import os
import pandas
import pandas as pd
import numpy as np
import pywt as pywt
import pickle
from BENDRmain.misc import * 
#from BoW.configs import *
from sklearn.cluster import KMeans, MiniBatchKMeans
from memory_profiler import profile
import torch
from kmeans_pytorch import kmeans
import cv2
import random

debug = True
@profile
def do_codebook(dataSource, abspath, codebook_size, sub_length, inter_point, max_iterations, verbosity, dataset='test'):
    output_folder = os.path.join(abspath, 'codebook', dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    noChannels = determine_no_channels_from_dataset(dataset)

    all_descriptors = []

    if dataSource == 'csv':
        for i in range(1, 6):
            with open(f'feature/{dataset}/data_tran_s{sub_length}_i{inter_point}.pkl', 'rb') as f:
                data_tran = pickle.load(f)

            all_descriptors = makeDescriptors(all_descriptors, data_tran)


    elif dataSource == 'pickle':
        all_descriptors = []
        for channelNo in range(0, 1):
            channelNo = 0 #Workaround
            for fileNo in range(1):
                with open(f'{abspath}/feature/{dataset}/data_tran_s{sub_length}_i{inter_point}_ch{channelNo}_{fileNo}.pkl', 'rb') as f:
                #with open(f'{abspath}/feature/{dataset}/data_tran_s{sub_length}_i{inter_point}_ch{channelNo}_1.pkl', 'rb') as f: #Workaround
                    data_tran = pickle.load(f)

                all_descriptors = makeDescriptors(all_descriptors, data_tran, fileNo)
        
        random.shuffle(all_descriptors)
        print(len(all_descriptors))
        print('Descriptors done')
        all_descriptors=np.float32(all_descriptors)
        
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, max_iterations, 1.0)
        print("Criteria defined")
        ret, label, cluster_centers = cv2.kmeans(all_descriptors, codebook_size, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)
        #kmeans = MiniBatchKMeans(n_clusters=codebook_size, batch_size=codebook_size // 100, max_iter=max_iterations, verbose=verbosity, precompute_distances=False).fit(all_descriptors)
        #kmeans = KMeans(n_clusters=codebook_size, max_iter=max_iterations, verbose=verbosity).fit(all_descriptors)
        #all_descriptors = torch.tensor(all_descriptors)
        #cluster_ids_x, cluster_centers = kmeans(X=all_descriptors, num_clusters=codebook_size, distance='euclidean', device=torch.device('cuda:0'))

        print('KMeans finished')
        with open(f'{output_folder}/kmean_s{sub_length}_i{inter_point}_c{codebook_size}.pkl', 'wb+') as f:
            #pickle.dump(kmeans.cluster_centers_, f)
            pickle.dump(cluster_centers, f)




def makeDescriptors(all_descriptors, data_tran, fileNo):
    #print(list(data_tran[0]))
    print(len(data_tran))
    
    #dd = np.transpose(data_tran[0])
    #print(dd.shape)
    if debug:
        for i in range(len(data_tran)):
            dataEntryIndex = i * (fileNo + 1)
            all_descriptors.extend(np.transpose(data_tran[dataEntryIndex]))
    else:
        #temp_index = list(np.random.permutation(np.linspace(0, len(data_tran) - 1, len(data_tran), dtype=int)))
        #for j in range(0, len(temp_index) // 20):
        #    all_descriptors.extend(data_tran[temp_index[j]])
        pass

    return all_descriptors
