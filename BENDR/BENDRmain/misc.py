import glob
import os
import argparse

def clearOutputFolder(output_folder):
    files = glob.glob(f'{output_folder}/*')
    for f in files:
        os.remove(f)

def parse_args_bow():
    parser = argparse.ArgumentParser(description="Trains the MLP")
    parser.add_argument('--dataset', default="downstream", help="Downstream or pretraining datasets")
    parser.add_argument('--sub-length', default=128, type=int,
                        help="The length of the 'word")
    parser.add_argument('--codebook-size', default=1000, type=int)
    parser.add_argument('--inter-point', default=4, type=int)
    parser.add_argument('--max-iterations', default=7, help="Maximum number of kmeans iterations")
    return parser.parse_args()

def parse_args_mlp():
    parser = argparse.ArgumentParser(description="Trains the MLP")
    parser.add_argument('--dataset', default="downstream", help="Downstream or pretraining datasets")
    parser.add_argument('--sub-length', default=128, type=int,
                        help="The length of the 'word")
    parser.add_argument('--codebook-size', default=1000, type=int)
    parser.add_argument('--inter-point', default=4, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    return parser.parse_args()

def determine_no_channels_from_dataset(dataset):
    if dataset == 'sleep-edf':
    	return 2
    else:
    	return 19
    	
def determine_no_of_data_files_in_dataset(dataset, abspath):
    data_files = os.listdir(os.path.join(abspath, 'pre_BoW_raw', f'pre_BoW_representation_{dataset}'))

    fileNos = []

    for f in data_files:
        f = f.split('.')[0]
        fileNos.append(int(f.split('_')[1]))

    print(f'Number of files: {max(fileNos)}')
    return max(fileNos)
    
