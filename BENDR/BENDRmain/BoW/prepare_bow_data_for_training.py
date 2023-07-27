import os.path
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from configs import *
from random import randrange, shuffle, random, randint

absolute_root = '/home/vinogradova/Documents/PycharmProjects/dn3-master/BENDRmain/'
dataset = 'sleep-edf'
calculateSubsegmentIndices = True
max_pred = 3
batch_size = 32

channels = []
#prepare the input data
for channelNo in range(noChannels):
    #centres
    closest_centres_file = os.path.join(absolute_root, 'BoW', 'closest_centres', dataset, f'closest_centres_s{sub_length}_i{inter_point}_ch{channelNo}.pkl')
    with open(closest_centres_file, 'rb') as f:
        closest_centres = pickle.load(f)
    channels.append(closest_centres)

centroids_file = os.path.join(absolute_root, 'BoW', 'codebook', dataset, f'kmean_s{sub_length}_i{inter_point}_c{codebook_size}_ch{channelNo}.pkl')
with open(centroids_file, 'rb') as f:
    centroids = pickle.load(f)

#Sequences and labels
with open(os.path.join(absolute_root, 'BoW', 'pre_BoW_raw', f'pre_BoW_representation_{dataset}', f'annotations_0_ch{channelNo}.pkl'), 'rb') as f:
    annotations = pickle.load(f)

class sleepEDFSentencesDataset(Dataset):
    def __init__(self, annotations, centroids, closest_centres1,closest_centres2):
        self.sequences1 = []
        self.sequences2 = []
        self.labels = []

        # create 'sentences'
        for sequence_ind in range(0, len(annotations) - 1):
            sentence_index_range = (annotations[sequence_ind][0], annotations[sequence_ind + 1][0])
            label = annotations[sequence_ind + 1][2]

            # calculate the indices of whole "word" segments

            tokenIndices = list(range(sentence_index_range[0], sentence_index_range[1], sub_length))
            tokenisedSentence1, tokenisedSentence2 = [], []
            try:
                for tokenIndex in tokenIndices:
                    corrCentroidIndex = tokenIndex // inter_point
                    corrCentroid1 = centroids[closest_centres1[corrCentroidIndex]]
                    corrCentroid2 = centroids[closest_centres2[corrCentroidIndex]]
                    tokenisedSentence1.append(corrCentroid1)
                    tokenisedSentence2.append(corrCentroid2)
                self.sequences1.append(tokenisedSentence1)
                self.sequences2.append(tokenisedSentence2)
                label = sequence_ind % 7 #TODO: Workaround
                self.labels.append(label)
            except IndexError:
                print('Index error')
                break


    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return len(self.labels)

    def __getitem__(self, item):
        """Given an index return an example from the position.

            Arguments:

              item (:obj:`int`):
                  Index position to pick an example to return.

            Returns:
              :obj:`Dict[str, str]`: Dictionary of inputs that are used to feed
              to a model.

            """
        channel1 = torch.flatten(torch.tensor(self.sequences1[item], dtype=torch.float))
        channel2 = torch.flatten(torch.tensor(self.sequences2[item], dtype=torch.float))
        return {'channel1': channel1,
                'channel2': channel2,
                'label': torch.tensor(self.labels[item], dtype=torch.long)}

class sleepEDFSentencesDatasetForTransformer(Dataset):
    def __init__(self, annotations, centroids, closest_centres1,closest_centres2):
        self.sequences1 = []
        self.sequences2 = []
        self.labels = []

        # create 'sentences'
        for sequence_ind in range(0, len(annotations) - 1):
            sentence_index_range = (annotations[sequence_ind][0], annotations[sequence_ind + 1][0])
            label = annotations[sequence_ind + 1][2]

            # calculate the indices of whole "word" segments

            tokenIndices = list(range(sentence_index_range[0], sentence_index_range[1], sub_length))
            tokenisedSentence1, tokenisedSentence2 = [], []
            try:
                for tokenIndex in tokenIndices:
                    corrCentroidIndex = tokenIndex // inter_point
                    corrCentroid1 = closest_centres1[corrCentroidIndex]
                    corrCentroid2 = closest_centres2[corrCentroidIndex]
                    tokenisedSentence1.append(corrCentroid1)
                    tokenisedSentence2.append(corrCentroid2)
                self.sequences1.append(tokenisedSentence1)
                self.sequences2.append(tokenisedSentence2)
                label = sequence_ind % 7 #TODO: Workaround
                self.labels.append(label)
            except IndexError:
                print('Index error')
                break


    def __len__(self):
        r"""When used `len` return the number of examples.

        """

        return len(self.labels)

    def __getitem__(self, item):
        """Given an index return an example from the position.

            Arguments:

              item (:obj:`int`):
                  Index position to pick an example to return.

            Returns:
              :obj:`Dict[str, str]`: Dictionary of inputs that are used to feed
              to a model.

            """
        channel1 = self.sequences1[item]
        channel2 = self.sequences2[item]
        return {'channel1': channel1,
                'channel2': channel2,
                'label': self.labels[item]}

def make_transformer_batch(dataset, additional_tokens, all_tokens, max_pred, batch_size):
    batch = []
    positive = negative = 0

    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(whole_dataset)), randrange(len(whole_dataset))

        tokens_a, tokens_b = dataset[tokens_a_index], dataset[tokens_b_index]

        input_ids = [additional_tokens['CLS']] + tokens_a['channel1'] + [additional_tokens['CHSEP']] + tokens_a['channel2'] \
                    + [additional_tokens['SEP']] + tokens_b['channel1'] + [additional_tokens['CHSEP']] + tokens_b['channel2'] + [additional_tokens['SEP']]
        segment_ids = [0] * (1 + len(tokens_a)*2 + 1 + 1) + [1] * (len(tokens_b)*2 + 1 + 1)

        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence

        cand_masked_pos = [i for i, token in enumerate(input_ids)
                          if token != additional_tokens['CLS'] and token != additional_tokens['SEP']]
        shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = additional_tokens['MASK']
            elif random() < 0.5:
                index = randint(0, codebook_size - 1)
                input_ids[pos] = all_tokens[index]

        if tokens_a_index + 1 == tokens_b_index:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    return batch



#create a dataset
annotations = annotations[0][0]
whole_dataset = sleepEDFSentencesDatasetForTransformer(annotations, centroids, closest_centres, closest_centres)
cls = 0
sep = 1 #separates different timestamps
chsep = 2 #separates different channels
mask = 3

special_tokens_dict = {'CLS': cls, 'SEP': sep, 'CHSEP': chsep, 'MASK': mask}

tokens_list = []
for tokenKey in special_tokens_dict.keys():
    tokens_list.append(special_tokens_dict[tokenKey])

tokens_list.extend(list(range(4, len(centroids)+4)))

batch = make_transformer_batch(whole_dataset, special_tokens_dict, tokens_list, max_pred, batch_size)