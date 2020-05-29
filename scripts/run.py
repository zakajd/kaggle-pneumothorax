import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('--inference', action="store_true")
parser.add_argument('--train', action="store_true")
parser.add_argument('--test', action="store_true")
parser.add_argument('--locally', action="store_true")
parser.add_argument('--fold', type=int, required=False, default=None)
args, unknownargs = parser.parse_known_args()
print("Unknown args: ", unknownargs)


folds = list(range(5)) if args.fold is None else [args.fold]
for fold in folds:

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    jobname = config_name
    commands = []

    if args.train:
        command = [
            'python',
            'train.py',
            "--config_file", args.config,
            "--root", "/import/AI-Dx_working/User/DomainAdaptation/kaggle-pneumothorax/data/interim",
            '--fold', str(fold),
            *unknownargs
        ]
        commands.append(command)
        jobname += '_train'

    if args.inference:
        command = [
            'python',
            'inference.py',
            "--config_path", os.path.join('data/logs', config_name, str(fold)),
            '--predict_val',
            *unknownargs
        ]
        commands.append(command)
        jobname += '_inf'

    if args.test:
        command = [
            'python',
            'test.py',
            "--config_path", os.path.join('data/logs', config_name, str(fold)),
            *unknownargs
        ]
        commands.append(command)
        jobname += '_test'

    if len(commands) == 0:
        print("No command!!!")
        exit()

    subprocess_input = ['./scripts/clusternode.sh'] + commands[0]
    for command in commands[1:]:
        subprocess_input.append('&&')
        subprocess_input.extend(command)

    if args.locally:
        subprocess_input = ['bash'] + subprocess_input
    else:
        subprocess_input = ['sbatch', '--job-name={}'.format(jobname)] + subprocess_input

    subprocess.run(subprocess_input)
