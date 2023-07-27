from BoW.do_representation3 import *
from BoW.do_codebook_tueg import *
from BoW.do_vq2 import *
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
    sub_length = args.sub_length
    codebook_size = args.codebook_size
    inter_point = args.inter_point
    max_iterations = 2
    verbosity = 1
    noChannels = 2
else:
    args = parse_args_bow()
    experiment = ExperimentConfig(f'./configs/downstream.yml')
    codebook_size = 1000
    max_iterations = 2
    verbosity = 1
    noChannels = 2

ds_name = list(experiment.datasets)[0]
print(ds_name)
@profile
def runDoAll():
    print('Triggering do representation')
    do_codebook(dataSource, absolute_root_bow, codebook_size, sub_length, inter_point, max_iterations, verbosity, ds_name)

runDoAll()
