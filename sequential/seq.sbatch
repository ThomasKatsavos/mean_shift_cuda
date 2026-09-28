#!/bin/bash
#SBATCH --job-name=mean-shift
#SBATCH --partition=ondemand
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=result-%j.log


srun ./mean_shift_gpu_serial photo2.jpeg result_128.png  128
