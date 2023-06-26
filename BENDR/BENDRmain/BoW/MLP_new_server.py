import os.path
import pickle
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import torch
#from configs import *
from random import randrange, shuffle, random, randint
from torchmetrics import Accuracy
from misc import *

train_batch_size = 128
lr = 1e-4
data_split = {'training': 0.70, 'validation': 0.25, 'testing': 0.05}

absolute_root = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain'
dataset = 'sleep-edf'
calculateSubsegmentIndices = True
max_pred = 3
batch_size = 128
server = True
noEpochs = 2

if server:
    args = parse_args_mlp()
    print(args)
    #experiment = ExperimentConfig(f'./configs/{args.dataset}.yml')
    sub_length = args.sub_length
    codebook_size = args.codebook_size
    inter_point = args.inter_point
    batch_size = args.batch_size
    noEpochs = args.epochs
    lr = args.lr
else:
    args = parse_args_mlp()
    #experiment = ExperimentConfig(f'./configs/downstream.yml')
    max_iterations = 2
    verbosity = 1
    noChannels = 2
    batch_size = 128
    lr = 1e-4
    sub_length = 256
    codebook_size = 750
    inter_point = 4
    
    
noChannels = 2

closest_centres = {'ch0': [], 'ch1': []}

bowroot = os.path.join(absolute_root, 'BoW')

def determine_no_of_data_files_in_dataset(dataset, abspath):
    data_files = os.listdir(os.path.join(abspath, 'BoW', 'pre_BoW_raw', f'pre_BoW_representation_{dataset}'))

    fileNos = []

    for f in data_files:
        f = f.split('.')[0]
        fileNos.append(int(f.split('_')[1]))

    print(f'Number of files: {max(fileNos)}')
    return max(fileNos)

noFiles= determine_no_of_data_files_in_dataset(dataset, absolute_root)
noFiles = 30
#prepare the input data
for channelNo in range(noChannels):
    #centres
    for fileNo in range(0, noFiles):
       closest_centres_file = os.path.join(absolute_root, 'BoW', 'closest_centres', dataset, f'closest_centres_s{sub_length}_i{inter_point}_c{codebook_size}_ch{channelNo}_{fileNo}_sent.pkl')
       with open(closest_centres_file, 'rb') as f:
           closest_centres_channel = pickle.load(f)
        
       closest_centres[f'ch{channelNo}'].extend(closest_centres_channel)
       


centroids_file = os.path.join(absolute_root, 'BoW', 'codebook', dataset, f'kmean_s{sub_length}_i{inter_point}_c{codebook_size}.pkl')
with open(centroids_file, 'rb') as f:
    centroids = pickle.load(f)

#Sequences and labels
annotations = []
for fileNo in range(0, noFiles):
    with open(os.path.join(absolute_root, 'BoW', 'pre_BoW_raw', f'pre_BoW_representation_{dataset}', f'annotations_{fileNo}.pkl'), 'rb') as f:
        annotations_local = pickle.load(f)
    for annotation in annotations_local:
        annotations.append(annotation[0])

class sleepEDFSentencesDataset(Dataset):
    def __init__(self, annotations_whole, centroids, closest_centres1_whole, closest_centres2_whole):
        self.sequences1 = []
        self.sequences2 = []
        self.labels = []
        
        #loop through each session
        for annotationsInd in range(len(annotations_whole)):
            #get the closest centres for this file
            annotations = annotations_whole[annotationsInd]
            closest_centres1 = closest_centres1_whole[annotationsInd]
            closest_centres2 = closest_centres2_whole[annotationsInd]
            
            # create 'sentences'
            for sequence_ind in range(0, len(annotations)-1):
                sentence_index_range = (annotations[sequence_ind][0], annotations[sequence_ind + 1][0])
                label = annotations[sequence_ind + 1][2] - 1
                    
                # calculate the indices of whole "word" segments

                tokenIndices = list(range(sentence_index_range[0]//sub_length, sentence_index_range[1]//sub_length))
                tokenisedSentence1, tokenisedSentence2 = [], []
                for tokenIndex in tokenIndices:
                    corrCentroid1 = centroids[closest_centres1[tokenIndex]]
                    corrCentroid2 = centroids[closest_centres2[tokenIndex]]
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
        channels = self.sequences1[item] + self.sequences2[item]
        concatChannels = torch.flatten(torch.tensor(channels, dtype=torch.float))
        return {'text': concatChannels,
                'label': torch.tensor(self.labels[item], dtype=torch.long)}

results_train = {'Epoch':[], 'Train_Loss': [], 'Train_Accuracy': [], 'Val_Loss': [], 'Val_Accuracy': []}
results_test = {'Loss': [], 'Accuracy': []}

#define the model
class Perceptron(torch.nn.Module):
    def __init__(self, no_of_tokens):
        super(Perceptron, self).__init__()
        self.fc = torch.nn.Linear(no_of_tokens, 1024)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(1024, 8)

    def forward(self, text, text_length):
        x = self.fc(text)
        x = self.relu(x)
        output = self.fc2(x)  # instead of Heaviside step fn
        return output


def accuracy(probs, target):
  predictions = probs.argmax(dim=1)
  corrects = (predictions == target)
  accuracy = corrects.sum().float() / float(target.size(1))
  return accuracy

def train(model, iterator, optimizer, criterion, epoch, epochs):
    print(f'Epoch {epoch}')
    epoch_loss = 0
    epoch_acc = 0
    batch_no = 0
    print(f"iterator length: {len(iterator)}")
    #loop = tqdm(iterator)
    #for idx, batch in enumerate(loop):
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch['text'], len(batch['text'])
        predictions = model(text, text_lengths)
        #target = torch.reshape(batch['label'], (batch['label'].shape[0], 1))
        
        loss = criterion(predictions, batch['label'])
        predictions_argmax = predictions.argmax(dim=1)
        accuracy = Accuracy(task="multiclass", num_classes=8)
        acc = accuracy(predictions_argmax, batch['label'])
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # add stuff to progress bar in the end
        #loop.set_description(f"Epoch [{epoch}/{epochs}]")
        #loop.set_postfix(loss=torch.rand(1).item(), acc=torch.rand(1).item())
        
        batch_no += 1
    
    epoch_avg_loss = epoch_loss / len(iterator)
    epoch_avg_acc = epoch_acc / len(iterator)
    results_train['Epoch'].append(epoch)
    results_train['Train_Loss'].append(epoch_avg_loss)
    results_train['Train_Accuracy'].append(epoch_avg_acc)
        
    print(f'Loss: {epoch_avg_loss}, accuracy: {epoch_avg_acc}')

    return epoch_avg_loss, epoch_avg_acc

def evaluate(model, iterator, criterion, valOrTest):
    print('Starting evaluation')
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch['text'], len(batch['text'])
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch['label'])
            accuracy = Accuracy(task="multiclass", num_classes=8)
            acc = accuracy(predictions, batch['label'])
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    final_loss = epoch_loss / len(iterator)
    final_acc = epoch_acc / len(iterator)
    
    if valOrTest == 'val':
        results_train['Val_Loss'].append(final_loss)
        results_train['Val_Accuracy'].append(final_acc)
    else:
        results_test['Loss'].append(final_loss)
        results_test['Accuracy'].append(final_acc)
    
    return final_loss, final_acc

def run_train(epochs, model, train_iterator, valid_iterator, test_iterator, optimizer, criterion, model_type, absolute_root):
    best_valid_loss = float('inf')
    
    results_path_folder = os.path.join(absolute_root, 'results', 'sleep-edf')

    for epoch in range(epochs):

        # train the model
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, epoch, epochs)

        # evaluate the model
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, 'val')
        
        
        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'saved_weights'+'_'+model_type+'.pt')
        
        results_filename_train = f'mlp_s{sub_length}_c{codebook_size}_batch{train_batch_size}_e{noEpochs}_lr{lr}_train.csv'
        pd.DataFrame(results_train).to_csv(os.path.join(results_path_folder, results_filename_train))

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    
    test_loss, test_acc = evaluate(model, test_iterator, criterion, 'test')
    
    results_filename_test = f'mlp_s{sub_length}_c{codebook_size}_batch{train_batch_size}_e{noEpochs}_lr{lr}_test.csv'
    pd.DataFrame(results_test).to_csv(os.path.join(results_path_folder, results_filename_test))

    print(f'\t Test Loss: {valid_loss:.3f} |  Test Acc: {valid_acc * 100:.2f}%')

#create a dataset
print(f"Length of annotations: {len(annotations)}")

whole_dataset = sleepEDFSentencesDataset(annotations, centroids, closest_centres['ch0'], closest_centres['ch1'])
torch_train_dataloader = DataLoader(whole_dataset, batch_size=train_batch_size, shuffle=True)
print(torch_train_dataloader.dataset)
print('Data loader created')

channels = whole_dataset.sequences1[0] + whole_dataset.sequences2[0]
concatChannels = torch.flatten(torch.tensor(channels, dtype=torch.float))

#split into training, validation and testing
training_length, val_length = int(data_split['training']*len(torch_train_dataloader)), \
                                              int(data_split['validation']*len(torch_train_dataloader))
testing_length = len(torch_train_dataloader) - training_length - val_length
training_data, validation_data, testing_data = random_split(torch_train_dataloader, [training_length, val_length, testing_length])
input_size = torch_train_dataloader.dataset[0]['text'].shape[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Perceptron(input_size)
loss_func = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr)

print('Training settings set up')

#train(model, training_data.dataset, optimiser, loss_func)
run_train(noEpochs, model, training_data.dataset, validation_data.dataset, testing_data.dataset, optimiser, loss_func, f'BoW_MLP_i{inter_point}_c{codebook_size}', absolute_root)

results_path_folder = os.path.join(absolute_root, 'results', 'sleep-edf')
print(results_path_folder)

if not os.path.exists(results_path_folder):
    os.makedirs(results_path_folder)

results_filename_train = f'mlp_s{sub_length}_c{codebook_size}_batch{train_batch_size}_e{noEpochs}_lr{lr}_train.csv'
pd.DataFrame(results_train).to_csv(os.path.join(results_path_folder, results_filename_train))

results_filename_test = f'mlp_s{sub_length}_c{codebook_size}_batch{train_batch_size}_e{noEpochs}_lr{lr}_test.csv'
pd.DataFrame(results_test).to_csv(os.path.join(results_path_folder, results_filename_test))


