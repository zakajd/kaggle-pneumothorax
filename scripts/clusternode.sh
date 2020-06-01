#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

## use it to set 1 gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=[1080ti]     # [1080ti|2080ti] critical!!! 2080ti does not suit!!!
#SBATCH --mem=45000

#SBATCH --partition=clusternodes

##SBATCH --exclude=nvidia29,nvidia30,nvidia21,nvidia22,nvidia23
##SBATCH --nodelist=nvidia31 #,nvidia30,nvidia29,nvidia28,nvidia26,nvidia25,nvidia24,nvidia23,nvidia22

#SBATCH -o /home/CODE1/320001019/workspace/logs/sbatch/clusternode/%x-%j-%N.out # STDOUT
#SBATCH -e /home/CODE1/320001019/workspace/logs/sbatch/clusternode/%x-%j-%N.err # STDERR

echo "$@"

bash << endl
$@
endl
