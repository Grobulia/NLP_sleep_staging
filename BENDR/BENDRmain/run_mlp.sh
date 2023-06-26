#!/bin/sh
#SBATCH --ntasks=60
#SBATCH --nodes=2
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=125000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=surname.name@uni-ulm.de

echo "Bash script is running"

export "PYTHONPATH=$PYTHONPATH:/pfs/work7/workspace/scratch/ul_hpx39-thesis/BENDR/"

echo "Pythonpath set"

source /pfs/data5/home/ul/ul_student/ul_hpx39/BENDR/BENDRmain/venv-python3/bin/activate

echo "venv activated"

sleep 30

python3 BoW/MLP_new_server.py --sub-length 256 --codebook-size 750 --epochs 100 --batch-size 128

deactivate
