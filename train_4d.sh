#! /bin/bash
#SBATCH -p sv-dgx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=24
#SBATCH --mem=36gb
#SBATCH --gpus-per-node=1
#SBATCH -J PWCnet
#SBATCH -o slurm_out/%J_out.txt
#SBATCH -e slurm_out/%J_err.txt

cd /home/liyang/workspace/baselines/PointPWC

which python

config="config_train_4dmatch.yaml"

python train.py ${config}