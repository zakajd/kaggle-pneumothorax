#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=tboard_ilyas
#SBATCH -o /import/AI-Dx_working/User/Ilyas/slurm_logs/tboard_%j.%N.out# STDOUT
#SBATCH -e /import/AI-Dx_working/User/Ilyas/slurm_logs/tboard_%j.%N.err# STDERR
#SBATCH --gres=gpu:0                   # GPU ressources
#SBATCH --nodelist=nvidia03
## get tunneling info
XDG_RUNTIME_DIR=""
path='/import/AI-Dx_working/User/Ilyas/aux_proj_data/DL_SSL/logs/'
ipnport=8893
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport user@host
    ----------------------------------------------------------------
"
#conda activate sl-ssl
## start an ipcluster instance and launch jupyter server
tensorboard --logdir=$path --port=$ipnport --bind_all
