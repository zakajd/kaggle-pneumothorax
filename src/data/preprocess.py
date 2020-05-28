"""
Convert RLE into binary masks, save images as PNG, apply clahe (optional)
Split data into train and validation with optional folds
"""
import os
import sys
import cv2
import glob
import time
import shutil
import pydicom
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.utils import shuffle
import pytorch_tools as pt

from pytorch_tools.utils.rle import rle_decode
from sklearn.model_selection import StratifiedKFold
# from multiprocessing import Pool
# from sklearn.model_selection import train_test_split

def main(hparams):
    train_dcms = sorted(glob.glob(hparams.root + '/siim/dicom-images-train/*/*/*.dcm'))
    test_dcms = sorted(glob.glob(hparams.root + '/siim/dicom-images-test/*/*/*.dcm'))

    # Read train labels and fix incorrect column name of CSV file
    df = pd.read_csv(hparams.root + '/siim/train-rle.csv', index_col=None)
    df.rename(columns={' EncodedPixels' : 'EncodedPixels'}, inplace=True)
    logger.info(f"Raw train images: {len(train_dcms)}, raw test images: {len(test_dcms)}, lines in DF: {len(df)}")

    # Used only in flag `use_clahe` is True
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    skipped = 0

    # Delete old images
    shutil.rmtree(hparams.output_path + '/images', ignore_errors=True)
    os.makedirs(hparams.output_path + '/images')

    if hparams.create_masks:
        shutil.rmtree(hparams.output_path + '/masks', ignore_errors=True)
        os.makedirs(hparams.output_path + '/masks')

    file_names, classes = [], []
    for idx in tqdm(range(len(train_dcms))):
        dataset = pydicom.dcmread(train_dcms[idx])
        img_id = train_dcms[idx].split('/')[-1][:-4]

        # Load image data
        pixels = clahe.apply(dataset.pixel_array) if hparams.use_clahe else dataset.pixel_array

        # Save as PNG only if image_id exists in DF
        if df[df['ImageId'] == img_id].empty:
            skipped += 1
            continue

        cv2.imwrite(f'{hparams.output_path}/images/{img_id}.png', pixels)

        rle_masks = df[df['ImageId'] == img_id]["EncodedPixels"].values
        if hparams.create_masks and (rle_masks[0] != '-1'):
            mask = np.zeros(pixels.shape)
            for rle in rle_masks:
                partial_mask = rle_decode(rle, pixels.shape, relative=True).astype(np.uint8)
                mask[partial_mask == 1] = 255
            if hparams.dilate_mask:
                kernel_sz = int(np.sqrt(np.sum(mask)) * 0.1)
                kernel = np.ones((kernel_sz, kernel_sz))
                mask = cv2.dilate(mask, kernel, iterations = 1)
            cv2.imwrite(f'{hparams.output_path}/masks/{img_id}.png', mask)
            pneumothorax = True
        elif hparams.create_masks:
            # Empty mask
            mask = np.zeros(pixels.shape)
            cv2.imwrite(f'{hparams.output_path}/masks/{img_id}.png', mask)
            pneumothorax = False

        if hparams.train_val_split:
            file_names.append(img_id)
            classes.append(int(pneumothorax))

    # Delete old test images
    shutil.rmtree(hparams.output_path + '/test_images', ignore_errors=True)
    os.makedirs(hparams.output_path + '/test_images')
    for idx in tqdm(range(len(test_dcms))):
        dataset = pydicom.dcmread(train_dcms[idx])
        img_id = train_dcms[idx].split('/')[-1][:-4]

        # Load image data
        pixels = clahe.apply(dataset.pixel_array) if hparams.use_clahe else dataset.pixel_array
        cv2.imwrite(f'{hparams.output_path}/test_images/{img_id}.png', pixels)

    logger.info(f'Finished generating images. Skiped {skipped} images')

    if hparams.train_val_split:
        logger.info("Started train/val split")
        file_names = np.array(file_names)
        classes = np.array(classes)

        final_splits = []

        # Add test split
        file_names, classes = shuffle(file_names, classes)
        train_size = int(hparams.train_size * len(file_names))
        test_size = len(file_names) - train_size

        test = np.vstack((file_names[:test_size],
                       classes[:test_size],
                       np.array(['test'] * test_size),
                       np.zeros(test_size))).T
        final_splits.append(test)

        # Add train/val splits
        file_names, classes = file_names[test_size:], classes[test_size:]
        skf = StratifiedKFold(n_splits=hparams.num_folds)
        for fold, (train_index, val_index) in enumerate(skf.split(file_names, classes)):
            files_train, files_val = file_names[train_index], file_names[val_index]
            classes_train, classes_val = classes[train_index], classes[val_index]
            train = np.vstack((files_train, 
                            classes_train, 
                            np.ones(files_train.shape, dtype=np.uint8) * fold, 
                            np.ones(files_train.shape, dtype=np.uint8))).T
            val = np.vstack((files_val, 
                            classes_val, 
                            np.ones(files_val.shape, dtype=np.uint8) * fold, 
                            np.zeros(files_val.shape, dtype=np.uint8))).T
            final_splits.append(train)
            final_splits.append(val)

        df_data = np.vstack(final_splits)
        # Save split into separate file for future usage
        df = pd.DataFrame(df_data, columns=['Index', 'Class', 'Fold', 'Train'])
        df.to_csv(f"{hparams.output_path}/train_val.csv", index=False)



if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Kaggle-Pneumothorax")
    parser.add_argument(
        "--root", type=str, default="data/raw", help="Path to all raw data as provided by organizers")
    parser.add_argument(
        "--output_path", type=str, default="data/interim", help="Path to save masks, PNGs and other files")
    parser.add_argument(
        "--train_size", type=float, default=0.8, help="Part of data used for training")
    parser.add_argument(
        "--create_masks", action="store_true", help="Flag to create masks for train images")
    parser.add_argument(
        "--train_val_split", action="store_true", help="Flag to devide data into train and validation parts")     
    parser.add_argument(
        "--num_folds", type=int, default=5, help="Number of splits for K-fold validation")     
    parser.add_argument(
        "--use_clahe", action="store_true", help="Flag to apply CLAHE to each image before saving") 
    parser.add_argument(
        "--dilate_mask", action="store_true", help="Flag to make mask slightly bigger")       

    # Setup logger
    logger.add(sys.stdout, format="{time:[MM-DD HH:mm:ss]} - {message}")

    hparams = parser.parse_args(sys.argv[1:])
    logger.info(f"Parameters used for preprocessing: {hparams}")

    start_time = time.time()
    main(hparams)
    logger.info(f"Finished preprocessing. Took: {(time.time() - start_time) / 60:.02f}m")