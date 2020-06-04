#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

#SBATCH --ntasks=1
#SBATCH --mem=20000

#SBATCH --partition=clusternodes

#SBATCH -o /home/CODE1/320001019/workspace/logs/sbatch/clusternode/%x-%j-%N.out # STDOUT
#SBATCH -e /home/CODE1/320001019/workspace/logs/sbatch/clusternode/%x-%j-%N.err # STDERR

echo "$@"

bash << endl
$@
endl
