#!/bin/bash
#
#SBATCH -N 1                           # number of nodes
#SBATCH -n 1                          # number of tasks
#SBATCH --mem=49152                    # memory per CPU
#SBATCH -p clusternodes                # partition
#SBATCH --gres=gpu:1                   # GPU ressources
#SBATCH --constraint=[1080ti]   # contraints
#SBATCH --exclude=nvidia01,nvidia04
#SBATCH --job-name=tube_segm
#SBATCH -o /import/AI-Dx_working/User/Ilyas/slurm_logs/%j.%N.out# STDOUT
#SBATCH -e /import/AI-Dx_working/User/Ilyas/slurm_logs/%j.%N.err# STDERR
#SBATCH --time=24:00:00
#SBATCH --begin=now

cd /import/AI-Dx_working/User/Ilyas/DL_SSL/
python /import/AI-Dx_working/User/Ilyas/DL_SSL/train_ssl.py -c $1