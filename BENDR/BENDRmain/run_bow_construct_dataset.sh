#!/bin/bash
#SBATCH --ntasks=50
#SBATCH --nodes=5
#SBATCH --time=48:00:00
##SBATCH --gres=gpu:1
#SBATCH --mem=150000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=surname.name@uni-ulm.de

echo "Bash script is running"

export "PYTHONPATH=$PYTHONPATH:/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/"

echo "Pythonpath set"

source /pfs/data5/home/ul/ul_student/ul_hpx39/BENDR/BENDRmain/venv-python3/bin/activate

echo "venv activated"

python3 do_all2.py
sleep 60
echo "Minikmeans precompute distances false"

python3 do_all_construct_dataset.py --dataset pretraining --sub-length 260 --codebook-size 750 --inter-point 4 #> out_listcomp.txt

#python3 do_all.py  --sub-length 260 --codebook-size 4124 --inter-point 4
#python3 do_all.py --sub-length 260 --codebook-size 4126 --inter-point 4 > out_kmeans_batch.txt


deactivate
