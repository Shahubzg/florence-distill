#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

module purge
module load profile/deeplrn
module load python/3.11.7
module load cuda/12.2

source ../env/venv/bin/activate

python ../repo/src/train.py
