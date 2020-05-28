import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--inference', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--locally', action="store_true")
parser.add_argument('--fold', type=int, required=False, default=None)
args = parser.parse_args()

folds = list(range(5)) if args.fold is None else [args.fold]
for fold in folds:
    if args.inference:
        job_name = os.path.basename(os.path.normpath(args.config))
        subprocess_input = [
            './scripts/clusternode.sh',
            'python',
            'inference.py',
            "--config_path", os.path.join(args.config, str(fold)),
            '--predict_val'
        ]
    elif args.test:
        job_name = os.path.basename(os.path.normpath(args.config))
        subprocess_input = [
            './scripts/clusternode.sh',
            'python',
            'test.py',
            "--config_path", os.path.join(args.config, str(fold)),
        ]
    else:
        job_name = os.path.splitext(os.path.basename(args.config))[0]
        subprocess_input = [
            './scripts/clusternode.sh',
            'python',
            'train.py',
            "--config_file", args.config,
            "--root", "/import/AI-Dx_working/User/DomainAdaptation/kaggle-pneumothorax/data/interim",
            '--fold', str(fold),
        ]

    if args.locally:
        subprocess_input = ['bash'] + subprocess_input
    else:

        subprocess_input = ['sbatch', '--job-name={}'.format(job_name)] + subprocess_input

    subprocess.run(subprocess_input)
