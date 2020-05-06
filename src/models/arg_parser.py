import os
import configargparse as argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Pneumothorax",
        default_config_files=["configs/default.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )

    add_arg = parser.add_argument

    # General
    add_arg("--name", type=str, help="Name of this run")
    add_arg("--seed", type=int, help="Random seed for reprodusability")
    add_arg("--root", type=str, help="Path to raw train data")
    add_arg("--batch_size", type=int, help="Batch size")
    add_arg("--workers", type=int, help="№ of data loading workers ")
    add_arg("--augmentation", default="light", type=str,help="How hard augs are")
    add_arg('--debug', dest='debug', default=False, action='store_true', help="Make short epochs")
    add_arg("--opt_level", default="O0", type=str, help="Optimization level for apex")
    add_arg("--resume", default="", type=str, help="Path to checkpoint to start from")
    add_arg('--fold', default=0, type=int, help='Number of fold to use for training')
    add_arg('--pos_weight', default=0.5, type=float, help="Proportion of positive examples sampled from dataset")

    # Model
    add_arg("--segm_arch", default="unet", type=str, help="Segmentation architecture to use")
    add_arg("--backbone", default="se_resnet50", help="Backbone architecture")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")

    # Training
    add_arg("--optim", type=str, default="adamw", help="Optimizer to use (default: adamw)")
    add_arg("--weight_decay", "--wd", default=1e-4, type=float, help="Weight decay (default: 1e-4)")
    add_arg("--size", default=512, type=int, help="Size of crops to train at")
    add_arg(
        "--phases",
        type=eval,
        action='append',
        help="Specify epoch order of data resize and learning rate schedule",
    )
    add_arg("--decoder_warmup_epochs", default=0, type=int, help="Number of epochs for training only decoder")
    add_arg(
        "--criterion", type=str, nargs="+", help="List of criterions to use. Should be like `bce 0.5 dice 0.5`",
    )

    # Validation and testing
    add_arg("--validate", dest='validate', default=False, action='store_true', 
        help="Flag to compute final metric on val set" )
    add_arg("--val_size", type=int, default=768, help="Predict on resized, then upscale")
    add_arg("--tta", dest='tta', default=False, action='store_true',
        help="Flag to use TTA for validation and test sets")
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()

    # If folder already exist append version number
    outdir = os.path.join("logs/", args.name)
    if os.path.exists(outdir):
        version = 1
        while os.path.exists(outdir):
            outdir = os.path.join("logs/", args.name + "_" + str(version))
            version += 1

    args.outdir = outdir
    return args