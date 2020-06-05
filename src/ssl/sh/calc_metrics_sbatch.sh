PATH_TO_SCRIPT='/import/AI-Dx_working/User/Ilyas/DL_SSL/src/ssl/sh/calc_metrics.sh'
sbatch "$PATH_TO_SCRIPT" --config_path "$1" --masks_path "$2" --output_path "$3" --min_iou 0.5
