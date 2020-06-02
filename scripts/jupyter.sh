#!/usr/bin/env bash


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=jupyter
#SBATCH --nodelist=nvidia19
#SBATCH --output /home/CODE1/320001019/workspace/logs/sbatch/interactive/jupyter-log-%j.%N.out # STDOUT
#SBATCH --error /home/CODE1/320001019/workspace/logs/sbatch/interactive/jupyter-log-%j.%N.err # STDERR


## get tunneling info

path="/home/CODE1/320001019/workspace/"
ipnport=8050
ipnip=$(hostname -i)
## print tunneling instructions to jupyter-log-{jobid}.txt
echo -e "
    Copy/Paste this in your local terminal to ssh tunnel with remote
    -----------------------------------------------------------------
    ssh -N -L $ipnport:$ipnip:$ipnport user@host
    ----------------------------------------------------------------
"


## start an ipcluster instance and launch jupyter server
jupyter lab  $path --no-browser --port=$ipnport --ip=$ipnip  --certfile=~/.jupyterlab_cert/jupyterlab.pem --keyfile=~/.jupyterlab_cert/jupyterlab.key
