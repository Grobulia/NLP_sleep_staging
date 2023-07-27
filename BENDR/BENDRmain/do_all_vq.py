from BoW.do_representation3 import *
from BoW.do_codebook3 import *
from BoW.do_vq_sentences import *
from BoW.BoW_data_loader import *
from misc import *
#from memory_profiler import profile

runNo = 1
dataSource = 'pickle'
absolute_root = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain'
absolute_root_bow = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain/BoW'
server = True
print('Initial script running')

if server:
    args = parse_args_bow()
    print(args)
    experiment = ExperimentConfig(f'./configs/{args.dataset}.yml')
    dataset = args.dataset
    sub_length = args.sub_length
    codebook_size = args.codebook_size
    inter_point = args.inter_point
    max_iterations = 2
    verbosity = 1
    noChannels = 2
else:
    args = parse_args_bow()
    experiment = ExperimentConfig(f'./configs/downstream.yml')
    dataset = "downstream"
    codebook_size = 1000
    max_iterations = 2
    verbosity = 1
    noChannels = 2

if dataset == 'downstream':
     ds_name = 'sleep-edf'
elif dataset == 'pretraining':
     ds_name = 'TUEG'

@profile
def runDoAll():
    #ds_name = 'sleep-edf'
    #do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 0, 0, ds_name)
    #do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 10, 0, ds_name)
    #do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 20, 0, ds_name)

    do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 22, 1, ds_name)
    #do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 10, 1, ds_name)
    #do_vq(runNo, absolute_root_bow, dataSource, codebook_size, sub_length, inter_point, max_iterations, 20, 1, ds_name)

runDoAll()
