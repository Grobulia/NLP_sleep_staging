from BoW.do_representation_tueg import *
from BoW.do_codebook3 import *
from BoW.do_vq2 import *
from BoW.BoW_data_loader import *
from misc import *
from memory_profiler import profile

runNo = 1
dataSource = 'pickle'
absolute_root = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain'
absolute_root_bow = '/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/BENDRmain/BoW'
server = True
print('Initial script running')

if server:
    args = parse_args_bow_tueg()
    print(args)
    experiment = ExperimentConfig(f'./configs/{args.dataset}.yml')
    sub_length = args.sub_length
    codebook_size = args.codebook_size
    inter_point = args.inter_point
    max_iterations = 2
    verbosity = 1
    noChannels = 2
    starting_file = args.starting_file
    final_file = args.final_file
else:
    #args = parse_args_bow()
    experiment = ExperimentConfig(f'./configs/downstream.yml')
    codebook_size = 1000
    max_iterations = 2
    verbosity = 1
    noChannels = 2

ds_name = list(experiment.datasets)[0]
print(ds_name)
#ds_name = 'sleep-edf'

#TODO: change to npy because otherwise things will overflow
#do_vq(runNo, absolute_root, dataSource)
@profile
def runDoAll():
    print('Triggering do representation')
    do_representation(dataSource, absolute_root_bow, codebook_size, sub_length, inter_point, max_iterations, dataset=ds_name, startingFileNo=starting_file, finalFileNo=final_file, channelNo=0)

    do_representation(dataSource, absolute_root_bow, codebook_size, sub_length, inter_point, max_iterations, dataset=ds_name, startingFileNo=starting_file, finalFileNo=final_file, channelNo=1)
    #do_representation(dataSource, absolute_root_bow, codebook_size, sub_length, inter_point, max_iterations, dataset=ds_name, startingFileNo=10, finalFileNo=20, channelNo=1)
    #do_representation(dataSource, absolute_root_bow, codebook_size, sub_length, inter_point, max_iterations, dataset=ds_name, startingFileNo=20, finalFileNo=32, channelNo=1)

runDoAll()
