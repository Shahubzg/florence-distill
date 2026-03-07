#!/bin/bash
#SBATCH --job-name=florence_baseline
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.err

module load profile/deeplrn
module load python/3.11.7

source /leonardo_work/IscrC_DEMOLLM/florence_distill/env/venv/bin/activate

cd /leonardo_work/IscrC_DEMOLLM/florence_distill/repo

bash scripts/run_baseline.sh
