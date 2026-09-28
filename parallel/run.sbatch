#!/bin/bash
#SBATCH --job-name=mean-shift
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --output=result-%j.log

module load gcc/14.2.0 cuda/12.6.2


srun ./mean_shift_gpu photo.jpeg result_gpu.png 512
