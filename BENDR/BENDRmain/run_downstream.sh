#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=300000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=surname.name@uni-ulm.de

mkdir -p results

source /pfs/data5/home/ul/ul_student/ul_hpx39/BENDR/BENDRmain/venv-python3/bin/activate


# Train LO/MSO from scratch
#python3 downstream.py linear --random-init --results-filename "results/linear_random_init.xlsx" > out_downstream_linear.txt
#python3 downstream.py BENDR --random-init --results-filename "results/BENDR_random_init.xlsx" > out_downstream_bendr.txt

# Train LO/MSO from checkpoint
#python3 downstream.py linear --results-filename "results/linear.xlsx" > out_downstream_linear_checkpoint.txt
python3 downstream.py BENDR --results-filename "results/BENDR.xlsx" > out_downstream_BENDR222.txt

# Train LO/MSO from checkpoint with frozen encoder
#python3 downstream.py linear --freeze-encoder --results-filename "results/linear_freeze_encoder.xlsx"
#python3 downstream.py BENDR --freeze-encoder --results-filename "results/BENDR_freeze_encoder.xlsx"

deactivate
