import os.path
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
#from torchtext.legacy.data import BucketIterator
import torch
#from globalVars import *
from random import randrange, shuffle, random, randint
from torchmetrics import Accuracy
from transformers import BertModel
from torch import nn

#TODO: automatic calculations for each dataset
sub_length = 256 #make sure the epoch is divisible by this number
inter_point = 4 #make sure it is divisible by the above, so that the segment indices make sense

codebook_size = 750
max_iterations = 2
verbosity = 1
noChannels = 2

absolute_root = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain'
dataset = 'sleep-edf'
calculateSubsegmentIndices = True
max_pred = 3
batch_size =  32
n_labels = 8
epochs = 40

channels = {'ch0': [], 'ch1': []}

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

device = torch.device('cuda') #if torch.cuda.is_available()# else torch.device('cpu')

train_batch_size = batch_size
lr = 0.001
data_split = {'training': 0.80, 'validation': 0.10, 'testing': 0.10}

dataset = 'sleep-edf'
noFiles = 31

closest_centres_all = {'ch0': [], 'ch1': []}

#prepare the input data
for channelNo in range(noChannels):
    #centres
    for fileNo in range(0, noFiles):
       closest_centres_file = os.path.join(absolute_root, 'BoW', 'closest_centres', 'sleep-edf_tueg', f'closest_centres_s{sub_length}_i{inter_point}_c{codebook_size}_ch{channelNo}_{fileNo}_sent.pkl')
       with open(closest_centres_file, 'rb') as f:
           closest_centres_channel = pickle.load(f)
        
       closest_centres_all[f'ch{channelNo}'].extend(closest_centres_channel)

centroids_file = os.path.join(absolute_root, 'BoW', 'codebook', 'tueg_v2_0', f'kmean_s{sub_length}_i{inter_point}_c{codebook_size}.pkl')
with open(centroids_file, 'rb') as f:
    centroids = pickle.load(f)

#Sequences and labels
annotations_all = []
for fileNo in range(0, noFiles):
    with open(os.path.join(absolute_root, 'BoW', 'pre_BoW_raw', f'pre_BoW_representation_{dataset}', f'annotations_{fileNo}.pkl'), 'rb') as f:
        annotations_local = pickle.load(f)
    annotations_all.extend(annotations_local)

standard_dict = {'Movement time': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3': 4,
    'Sleep stage 4': 5,
    'Sleep stage ?': 6,
    'Sleep stage R': 7,
    'Sleep stage W': 8}

annotations = []
#correct annotations
for index in range(len(annotations_all)):
    local_dictionary = annotations_all[index][1]
    data = annotations_all[index][0]
    mapping = dict([(local_dictionary[entry], standard_dict[entry]) for entry in local_dictionary])
    annotations.append([[data[entry][0], data[entry][1],  mapping[data[entry][2]]] for entry in range(len(data))])

class sleepEDFSentencesDatasetForTransformer(Dataset):
    def __init__(self, annotations_whole, startingInd, finalInd, centroids, closest_centres1_whole, closest_centres2_whole):
        self.sequences1 = []
        self.sequences2 = []
        self.labels = []
        
        #loop through each session
        for annotationsInd in range(startingInd, finalInd):
            #get the closest centres for this file
            annotations = annotations_whole[annotationsInd]
            closest_centres1 = closest_centres1_whole[annotationsInd]
            closest_centres2 = closest_centres2_whole[annotationsInd]
            
            # create 'sentences'
            for sequence_ind in range(0, len(annotations)-1):
                sentence_index_range = (annotations[sequence_ind][0], annotations[sequence_ind + 1][0])
                label = annotations[sequence_ind][2] - 1
                    
                # calculate the indices of whole "word" segments
                tokenIndices = range(sentence_index_range[0]//sub_length, sentence_index_range[1]//sub_length)
                tokenisedSentence1, tokenisedSentence2 = [], []
                for tokenIndex in tokenIndices:
                    corrCentroid1 = closest_centres1[tokenIndex]
                    corrCentroid2 = closest_centres2[tokenIndex]
                    tokenisedSentence1.append(corrCentroid1)
                    tokenisedSentence2.append(corrCentroid2)
                self.sequences1.append(tokenisedSentence1)
                self.sequences2.append(tokenisedSentence2)
                self.labels.append(label)



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


def make_transformer_inputs_classification(dataset, additional_tokens, all_tokens, max_pred):
    batch = []
    inputs = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
    sampleSize = len(dataset)
    for sampleNo in range(sampleSize):
        tokens_a = dataset[sampleNo]

        input_ids = [additional_tokens['CLS']] + tokens_a['channel1'] + [additional_tokens['CHSEP']] + tokens_a['channel2'] + [additional_tokens['SEP']]
        segment_ids = [0] * (1 + len(tokens_a['channel1']) + 1 + len(tokens_a['channel2']) + 1)

        input_ids_unmasked = input_ids.copy()
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # 15 % of tokens in one sentence
        inputs['input_ids'].append(input_ids)
        inputs['token_type_ids'].append(segment_ids)
        inputs['attention_mask'].append([0]*len(input_ids))
        inputs['labels'].append(tokens_a['label'])
        
    inputs['ids'] = torch.tensor(inputs['input_ids'])
    inputs['token_type_ids'] = torch.tensor(inputs['token_type_ids'])
    inputs['attention_mask'] = torch.tensor(inputs['attention_mask'])
    inputs['labels'] = torch.tensor(inputs['labels'])

    return inputs

class InputDatasetForTransformer(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}

    def __len__(self):
        return len(self.inputs['input_ids'])

annotations_training_length = len(annotations)

training_length, val_length = int(data_split['training']*annotations_training_length), int(data_split['validation']*annotations_training_length)
testing_length = annotations_training_length - training_length - val_length



training_start_index, training_final_index = 0, training_length
val_start_index, val_final_index = training_final_index, training_final_index + val_length
test_start_index, test_final_index = val_final_index, val_final_index + testing_length


training_dataset = sleepEDFSentencesDatasetForTransformer(annotations, training_start_index, training_final_index, centroids, closest_centres_all['ch0'], closest_centres_all['ch1'])
val_dataset = sleepEDFSentencesDatasetForTransformer(annotations, val_start_index, val_final_index, centroids, closest_centres_all['ch0'], closest_centres_all['ch1'])
test_dataset = sleepEDFSentencesDatasetForTransformer(annotations, test_start_index, test_final_index, centroids, closest_centres_all['ch0'], closest_centres_all['ch1'])


cls = 0
sep = 1 #separates different timestamps
chsep = 2 #separates different channels
mask = 3

special_tokens_dict = {'CLS': cls, 'SEP': sep, 'CHSEP': chsep, 'MASK': mask}

tokens_list = []
for tokenKey in special_tokens_dict.keys():
    tokens_list.append(special_tokens_dict[tokenKey])

tokens_list.extend(list(range(4, len(centroids)+4)))
sampleSize = len(training_dataset)

training_input = make_transformer_inputs_classification(training_dataset, special_tokens_dict, tokens_list, max_pred)
validation_input = make_transformer_inputs_classification(val_dataset, special_tokens_dict, tokens_list, max_pred)
test_input = make_transformer_inputs_classification(test_dataset, special_tokens_dict, tokens_list, max_pred)

trainingInputDataset = InputDatasetForTransformer(training_input)
validationInputDataset = InputDatasetForTransformer(validation_input)
testInputDataset = InputDatasetForTransformer(test_input)

training_loader = torch.utils.data.DataLoader(trainingInputDataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validationInputDataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(testInputDataset, batch_size=batch_size, shuffle=True)



class BERTLargeClass(torch.nn.Module):
    def __init__(self):
        super(BERTLargeClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, n_labels)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.l2(pooled_output)
        output = self.l3(pooled_output)
        return output

device = torch.device('cuda') #if torch.cuda.is_available()# else torch.device('cpu')
       
        
modelpath = os.path.join(absolute_root, f'saved_weights_BERT_b32_lr0.0001.h5') 
model = BERTLargeClass()
#model.to(device)
model.load_state_dict(torch.load(modelpath),strict=False)
model.to(device)

criterion = nn.CrossEntropyLoss()
###############################################################################
# Create DataLoader and Model
###############################################################################

train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0
                }



optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)
#model.to(device)

###############################################################################
# Train Model
###############################################################################

results_train = {'Epoch':[], 'Train_Loss': [], 'Train_Accuracy': [], 'Val_Loss': [], 'Val_Accuracy': []}

results_path_folder = os.path.join(absolute_root, 'results', 'sleep-edf')
print(results_path_folder)

if not os.path.exists(results_path_folder):
    os.makedirs(results_path_folder)

def train_model(start_epochs, n_epochs, training_loader, validation_loader, model,optimizer):
  
  loss_vals = []
  print(n_epochs)
  for epoch in range(start_epochs, n_epochs+1):
    train_loss = 0
    valid_loss = 0
    
    train_acc = 0
    valid_acc = 0

    ######################    
    # Train the model #
    ######################

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    for batch_idx, data in enumerate(training_loader):
        optimizer.zero_grad()
        #Forward
        ids = data['ids'].to(device, dtype=torch.long)
        targets = data['labels'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        outputs = model(ids, mask,token_type_ids)
       
        accuracy = Accuracy(task="multiclass", num_classes=8).to(device)
        train_acc = train_acc + accuracy(outputs, targets).cpu().numpy()
        #Backward
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))


    ######################    
    # Validate the model #
    ######################

    model.eval()
    with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                targets = data['labels'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                outputs = model(ids, mask, token_type_ids)
                accuracy = Accuracy(task='multiclass', num_classes=8).to(device)
                valid_acc = valid_acc + accuracy(outputs, targets).cpu().numpy()
                #results_train['Epoch'].append(epoch)
                #results_train['Train_Loss'].append(train_loss)
                #results_train['Train_Accuracy'].append(train_acc)
                loss = criterion(outputs, targets)
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))

            # calculate average losses
            train_loss = train_loss / len(training_loader)
            valid_loss = valid_loss / len(validation_loader)
            train_acc = train_acc / len(training_loader)
            valid_acc = valid_acc / len(validation_loader)
            # print training/validation statistics
            print('Epoch: {} \tAverage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}, Train Accuracy: {:.6f} \t Val Accuracy: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss,
                train_acc,
                valid_acc
              ))
            loss_vals.append(valid_loss)
            results_train['Epoch'].append(epoch)
            results_train['Train_Loss'].append(train_loss)
            results_train['Train_Accuracy'].append(train_acc)
            results_train['Val_Loss'].append(valid_loss)
            results_train['Val_Accuracy'].append(valid_acc)
            results_filename_train = f'bert_s{sub_length}_c{codebook_size}_batch{train_batch_size}_e{n_epochs}_lr{lr}_split_train.csv'
            pd.DataFrame(results_train).to_csv(os.path.join(results_path_folder, results_filename_train))

  return model

trained_model = train_model(1, 40, training_loader, validation_loader, model, optimizer)
print('Saving model')
torch.save(trained_model.state_dict(), f'{absolute_root}/saved_weights_bert_fine-tuned_b{batch_size}_{lr}.h5')

 

