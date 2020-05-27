import argparse
import time
import os
import yaml
import skimage.io
import skimage.transform
import numpy as np
import tqdm
from sklearn.utils import shuffle

from src.data.datasets import PneumothoraxDataset


def visualize(hparams):
    assert os.path.exists(hparams.config_path)
    with open(os.path.join(hparams.config_path, "config.yaml"), "r") as file:
        model_configs = yaml.load(file)
    vars(hparams).update(model_configs)

    dataset = PneumothoraxDataset(
        root=hparams.root,
        fold=hparams.fold,
        train=False,
    )
    image_paths, gt_paths = shuffle(dataset.images, dataset.masks)

    pred_masks_dir = os.path.join(hparams.masks_path, "val_prediction", hparams.name)

    output_dir = os.path.join(hparams.output_path, hparams.name)
    os.makedirs(output_dir, exist_ok=True)

    i = 0
    for image_path, gt_path in tqdm.tqdm(zip(image_paths, gt_paths)):
        image_name = os.path.basename(image_path)

        image = skimage.io.imread(image_path)
        gt = skimage.io.imread(gt_path)
        pred = skimage.io.imread(os.path.join(pred_masks_dir, image_name))
        pred = (skimage.transform.resize(pred, gt.shape) > 0.5) * 255.

        overlay = np.stack((pred, gt, np.zeros(pred.shape)), axis=2)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        overlay = 0.7 * image + 0.3 * overlay
        overlay = np.concatenate((overlay, image), axis=1)
        skimage.io.imsave(os.path.join(output_dir, image_name), overlay.astype(np.uint8))
        i += 1
        if i >= hparams.n:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIIM challenge",
    )

    parser.add_argument(
        "--config_path", type=str, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--masks_path", type=str, default="data/processed", help="Path to save masks")
    parser.add_argument(
        "--output_path", type=str, default="data/overlay", help="Path to save overlays")
    parser.add_argument(
        "--n", type=int, default=20, help="Number of random images to visualize")

    hparams = parser.parse_args()
    print(f"Parameters used for inference: {hparams}")
    start_time = time.time()
    visualize(hparams)
    print(f"Finished inference. Took: {(time.time() - start_time) / 60:.02f}m")