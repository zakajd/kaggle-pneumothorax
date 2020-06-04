# Predict test images using TTA
# Save individual models predictions
# Enseble them with Max vote or simple average

import os
import time
import shutil
from pathlib import Path
from multiprocessing import pool
import configargparse as argparse

import cv2
import yaml
import apex
import torch
import shapely
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_tools as pt
import skimage.io
import albumentations as albu
import skimage.transform

# Local imports
from src.data.datasets import get_test_dataloader, get_val_dataloader
from src.utils import MODEL_FROM_NAME
from src.callbacks import ThrJaccardScore
from test import delete_small_regions, label_mask


@torch.no_grad()
def predict_from_loader(model, loader, file_names, output_path):
    """Predict, threshold, save
    Args:
        model
        loader
        file_names (list)
        output_path (str)
    """
    idx = 0
    for batch in tqdm(loader):
        # Take images if (images, masks) returned
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        prediction = model(batch).sigmoid().cpu()
        for p in prediction:
            p = (p[0].squeeze().numpy() * 255).astype(np.uint8)
            skimage.io.imsave(os.path.join(output_path, file_names[idx]), p)
            idx += 1


def mask2rle(img, width, height):
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel+=1
    return " ".join(rle)


def buid_submission(rle_dict, sample_sub):
    sub = pd.DataFrame.from_dict([rle_dict]).T.reset_index()
    sub.columns = sample_sub.columns
    sub.loc[sub.EncodedPixels == '', 'EncodedPixels'] = -1
    return sub


@torch.no_grad()
def submit_from_loader(model, loader, file_names,  threshold_prob, threshold_area,
                       sample_sub):
    """Predict, threshold, center crop (to 900 x 900 size), save
    Args:
        model
        loader
        file_names (list)
        output_path (str)
    """
    idx = 0
    rle_dict = {}
    for batch in tqdm(loader):
        # Take images if (images, masks) returned
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # B x 4 x H x W -> B x H x W
        prediction = model(batch).sigmoid().cpu().squeeze()

        for p in prediction:
            p = p.numpy()
            mask = p > threshold_prob
            labeled_mask, num = label_mask(mask)
            labeled_mask, _ = delete_small_regions(labeled_mask, num, threshold_area)
            mask = labeled_mask > 0
            mask = skimage.transform.resize(mask, (1024, 1024)).astype(int) * 255
            rle_dict[os.path.splitext(file_names[idx])[0]] = mask2rle(mask.T, 1024, 1024)
            idx += 1

    sub = buid_submission(rle_dict, sample_sub)
    return sub


def main(hparams):
    assert os.path.exists(hparams.config_path)
    # Add model parameters 
    with open(os.path.join(hparams.config_path, "config.yaml"), "r") as file:
        model_configs = yaml.load(file)
        model_configs['use_jsrt_china_dataset'] = model_configs.get('use_jsrt_china_dataset', False)
    model_configs.update(vars(hparams))
    hparams = model_configs = argparse.Namespace(**model_configs)

    print("Loading model")
    model = MODEL_FROM_NAME[hparams.segm_arch](hparams.backbone, num_classes=(1 + hparams.use_jsrt_china_dataset),
                                               **hparams.model_params)

    # Convert all Conv2D -> WS_Conv2d if needed
    if hparams.ws:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model).cuda()

    checkpoint = torch.load(os.path.join(hparams.config_path , "model.chpn"))

    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda().eval()

    # if FLAGS.tta:
    #     model = pt.tta_wrapper.TTA(
    #         model, segm=True, h_flip=True, h_shift=[10, -10], v_shift=[10, -10],
    #         merge="gmean", activation="sigmoid"
    #     )

    model = apex.amp.initialize(model, verbosity=0)
    print("Model loaded succesfully")

    if hparams.predict_val:
        val_loader, val_files = get_val_dataloader(
            root=hparams.root,
            fold=hparams.fold,
            size=hparams.val_size,
            batch_size=hparams.batch_size,
            workers=hparams.workers
        )

        output_path = os.path.join(hparams.output_path, "val_prediction", hparams.name)
        print(f"Saving validation masks to {output_path}")

        # Delete old masks if any
        os.makedirs(output_path, exist_ok=True)

        # Shorten TIFF file name to <timestamp>-<timestamp>-tile-<tile_num>
        val_names = [os.path.basename(p) for p in val_files]
        predict_from_loader(model, val_loader, val_names, output_path)

    if hparams.predict_hold_out_test:
        val_loader, val_files = get_val_dataloader(
            root=hparams.root,
            fold="test",
            size=hparams.val_size,
            batch_size=hparams.batch_size,
            workers=hparams.workers
        )

        output_path = os.path.join(hparams.output_path, "hold_out_test_prediction", hparams.name, str(hparams.fold))
        print(f"Saving validation masks to {output_path}")

        # Delete old masks if any
        os.makedirs(output_path, exist_ok=True)

        # Shorten TIFF file name to <timestamp>-<timestamp>-tile-<tile_num>
        val_names = [os.path.basename(p) for p in val_files]
        predict_from_loader(model, val_loader, val_names, output_path)

    if hparams.predict_test:
        # Get dataloader
        test_loader, test_files = get_test_dataloader(
            root=hparams.root,
            size=hparams.val_size,
            batch_size=hparams.batch_size,
            workers=hparams.workers
        )

        output_path = os.path.join(hparams.output_path, "test_prediction", hparams.name)
        print(f"Saving test masks to {output_path}")

        # Delete old masks if any
        os.makedirs(output_path, exist_ok=True)

        # Shorten TIFF file name to <timestamp>-<timestamp>-tile-<tile_num>
        test_names = [os.path.basename(p) for p in test_files]

        if hparams.create_solution:
            sample_sub =  pd.read_csv(hparams.sample_sub_path)
            test_pred_df = submit_from_loader(model, test_loader, test_names,
                                              hparams.threshold_prob,  hparams.threshold_area, sample_sub)
            test_pred_df.to_csv(f"{hparams.output_path}/{hparams.name}_solution.csv", index=False)
        else:
            predict_from_loader(model, test_loader, test_names, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SIIM challenge",
    )

    parser.add_argument(
        "--config_path", type=str, help="Path to folder with model config and checkpoint")
    parser.add_argument(
        "--output_path", type=str, default="data/processed", help="Path to save masks")
    parser.add_argument(
        "--predict_val", action="store_true", help="Flag to make prediction for validation")
    parser.add_argument(
        "--predict_hold_out_test", action="store_true", help="Flag to make local test (hold out from train split of competition)")
    parser.add_argument(
        "--predict_test", action="store_true", help="Flag to make prediction for test")
    parser.add_argument(
        "--create_solution", action="store_true", help="Flag to transform masks into CSV")
    parser.add_argument(
        "--threshold_prob", type=float, default=None, help="")
    parser.add_argument(
        "--threshold_area", type=int, default=None, help="")
    parser.add_argument(
         "--sample_sub_path", type=str)

    hparams = parser.parse_args()
    print(f"Parameters used for inference: {hparams}")
    start_time = time.time()
    main(hparams)
    print(f"Finished inference. Took: {(time.time() - start_time) / 60:.02f}m")