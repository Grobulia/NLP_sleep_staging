import os
import pandas as pd
import numpy as np
import pywt as pywt
import pickle
import feather
import asyncio
from BENDRmain.misc import *
from memory_profiler import profile
import functools

print('do_representation accessed2')
#from BoW.configs import *

'''
original data = [1 channel] x [100 samples] x [4096 points]
'''
@profile
def do_representation(dataSource, abspath, codebook_size, sub_length, inter_point, max_iterations, dataset='test', startingFileNo=0, finalFileNo=0, channelNo=0):
    print('Starting do_representation')
    output_subfolder = f'feature/{dataset}'
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    # else:
    #     if startingFileNo == 0:
    #         clearOutputFolder(f'feature/{dataset}')

    print('Starting do_representation2')

    data_tran = {}
    indices = {}
    
    noChannels = determine_no_channels_from_dataset(dataset)
    noFiles = determine_no_of_data_files_in_dataset(dataset, abspath)
    #import the data
    if dataSource == 'csv':
        for i in range(1, 6):  # files
            data = pd.read_csv(os.path.join('csvdata', 'Data_%i.csv' % i), header=None)
            dataColumns = list(data.columns)

            for column in dataColumns:
                dataVector = list(data[column])

                data_tran[column] = makeFeatureVectors(dataVector, sub_length, inter_point)

    elif dataSource == 'pickle':

            print('Loading data')
            
            
            print(f'Channel {channelNo}:')
            
            if finalFileNo == 0:
                finalFileNo = noFiles

            for fileNo in range(startingFileNo, finalFileNo):
                print(f'  File: {fileNo}')

                data_tran, indices = {}, {}

                fileData = pd.read_feather(os.path.join(abspath, 'pre_BoW_raw', f'pre_BoW_representation_{dataset}_feather', f'prebow_{fileNo}_ch{channelNo}.feather')).to_dict()

                print('Data loaded')
                data = fileData['data']
                #print(data)
                fileData = []


                coroutines = []
                for entryIndex in range(0, len(data)):
                #for entryIndex in range(0, 1):
                    print(f"\n \n    entry {entryIndex}/{len(data)}")
                    dataEntryIndex = entryIndex * (fileNo + 1)
                    data_tran[dataEntryIndex], indices[dataEntryIndex] = makeFeatureVectors(sub_length, inter_point, data, fileNo, entryIndex)
                    #coroutines.append(makeFeatureVectors(sub_length, inter_point, data, fileNo, entryIndex))
                #results = await asyncio.gather(*coroutines)
                #print(results)
                data = []

                print('Saving')
                
            #all_participants_df = pd.DataFrame(data=data_tran).reset_index()
            #all_participants_df.to_feather(f'{output_folder}/{output_subfolder}/data_tran_s{sub_length}_i{inter_point}_ch{channelNo}.feather')
                with open(f'{abspath}/{output_subfolder}/data_tran_s{sub_length}_i{inter_point}_ch{channelNo}_{fileNo}.pkl', 'wb+') as f:
                    pickle.dump(data_tran, f, protocol=4)
                    f.close()



                #with open(f'{abspath}/{output_subfolder}/indices_s{sub_length}_i{inter_point}_{entryIndex}.pkl', 'wb+') as f:
                #    pickle.dump(indices, f)



@profile
def makeFeatureVectors(sub_length, inter_point, data, fileNo, entryIndex):
    dataVector = data[entryIndex]
    dataEntryIndex = entryIndex * (fileNo + 1)
    tran = []
    whole_segment_indices = []
    #iterations = list(range(0, len(dataVector) - sub_length, inter_point))
    #print(f'Number of iterations: {len(iterations)}')
    #tran = map(functools.partial(calculateDWTForSegment, sub_length, dataVector), iterations)
    #tran = [calculateDWTForSegment(sub_length, dataVector, i) for i in iterations]
    #for windowIndex in range(0, len(dataVector) - sub_length, inter_point):
    
    sub_sequences = np.array([dataVector[i:i+sub_length] for i in range(0, len(dataVector) - sub_length, inter_point)])
    print(sub_sequences)
    sub_sequences = (sub_sequences - np.mean(sub_sequences, axis=1, keepdims=True)) / np.std(sub_sequences, axis=1, ddof=1, keepdims=True)
    tran = np.transpose(pywt.wavedec(sub_sequences, 'db3', level=1, axis=1)[0])
    #print(tran)
    return tran, whole_segment_indices





