SH_PATH='/import/AI-Dx_working/User/Ilyas/DL_SSL/src/ssl/sh/infer.sh'
OUT_PATH='/import/AI-Dx_working/User/Ilyas/aux_proj_data/DL_SSL/preds_cons/'
sbatch "$SH_PATH" --config_path "$1" --output_path "$OUT_PATH" "$2"
