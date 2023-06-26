#!/bin/bash
#SBATCH --ntasks=70
#SBATCH --nodes=2
#SBATCH --time=6:00:00
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

#python3 do_all_representation.py  --sub-length 260 --codebook-size 4124 --inter-point 4
#python3 do_all_representation.py  --sub-length 360 --codebook-size 4124 --inter-point 4
#python3 do_all_representation.py  --sub-length 420 --codebook-size 4124 --inter-point 4

#python3 do_all_representation.py  --sub-length 256 --codebook-size 4124 --inter-point 4

#python3 do_all_representation.py  --sub-length 320 --codebook-size 4124 --inter-point 4

#python3 do_all_representation_tueg.py  --sub-length 256 --codebook-size 4124 --inter-point 4

#python3 do_all_representation_tueg.py  --sub-length 320 --codebook-size 4124 --inter-point 4

python3 do_all_representation_tueg.py  --dataset "pretraining" --sub-length 256 --codebook-size 750 --inter-point 4

deactivate
