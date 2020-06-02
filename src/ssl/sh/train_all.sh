#!/bin/bash
c_path="$1*"

for file in $c_path
do
    if [[ -f $file ]]; then
        sbatch /import/AI-Dx_working/User/Ilyas/DL_SSL/src/ssl/sh/train.sh $file &
    fi
done