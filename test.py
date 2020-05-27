import argparse
import time
import os
import yaml
import skimage.io
import skimage.transform
import numpy as np
import pandas as pd
import tqdm


from src.data.datasets import PneumothoraxDataset

def dice(gt, pred):
    gt = np.asarray(gt).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    if gt.shape != pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    if gt.any() or pred.any():
        intersection = np.logical_and(gt, pred)
        return 2. * intersection.sum() / (gt.sum() + pred.sum())
    else:
        return 1


def test(hparams):
    assert os.path.exists(hparams.config_path)
    with open(os.path.join(hparams.config_path, "config.yaml"), "r") as file:
        model_configs = yaml.load(file)
    vars(hparams).update(model_configs)

    pred_masks_dir = os.path.join(hparams.masks_path, "val_prediction", hparams.name)

    dataset = PneumothoraxDataset(
        root=hparams.root,
        fold=hparams.fold,
        train=False,
    )

    dices = {}
    for gt_mask_path in tqdm.tqdm(dataset.masks):
        image_name = os.path.basename(gt_mask_path)
        gt = skimage.io.imread(gt_mask_path)
        pred = skimage.io.imread(os.path.join(pred_masks_dir, image_name))
        pred = skimage.transform.resize(pred, gt.shape) > 0.5
        dices[image_name] = dice(gt, pred)

    dices = pd.DataFrame.from_dict(dices, orient='index', columns=['dice'])
    dices.loc['MEAN'] = dices.mean(axis=0)
    print(dices.loc['MEAN'])

    output_dir = os.path.join(hparams.output_path, hparams.name)
    os.makedirs(output_dir, exist_ok=True)
    dices.to_csv(os.path.join(output_dir, '{}_dice.csv'.format(hparams.fold)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIIM challenge",
    )

    parser.add_argument(
        "--config_path", type=str, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--masks_path", type=str, default="data/processed", help="Root path with masks")
    parser.add_argument(
        "--output_path", type=str, default="data/scores", help="Path to save scores")

    hparams = parser.parse_args()
    print(f"Parameters used for test: {hparams}")
    start_time = time.time()
    test(hparams)
    print(f"Finished test. Took: {(time.time() - start_time) / 60:.02f}m")