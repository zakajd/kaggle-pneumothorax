import argparse
import time
import os
import yaml
import skimage.io
import skimage.transform
import skimage.morphology
import skimage.measure
import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import confusion_matrix

from src.data.datasets import PneumothoraxDataset


def dice(gt, pred):
    gt = np.asarray(gt).astype(np.bool)
    pred = np.asarray(pred).astype(np.bool)

    if gt.shape != pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    gt_is_zero = not gt.any()
    pred_is_zero = not pred.any()

    if gt_is_zero and pred_is_zero:
        return 1
    elif gt_is_zero or pred_is_zero:
        return 0
    else:
        intersection = np.logical_and(gt, pred)
        return 2. * intersection.sum() / (gt.sum() + pred.sum())


def iou_score(mask1, mask2):
    return np.logical_and(mask1, mask2).sum() / np.logical_or(mask1, mask2).sum()


def label_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    dilated_mask = skimage.morphology.binary_dilation(mask, np.ones((10, 10)))
    labeled, num = skimage.measure.label(dilated_mask, connectivity=2, return_num=True)
    return labeled * mask, num


def calculate_region_areas(labeled_mask, num):
    return [(labeled_mask == label).sum() for label in range(1, num + 1)]


def delete_small_regions(labeled_mask, num, threshold_area, regions_areas=None):
    if regions_areas is None:
        regions_areas = calculate_region_areas(labeled_mask, num)

    relabeled_mask = np.zeros(labeled_mask.shape)
    new_label = 1
    for label, area in zip(range(1, num + 1), regions_areas):
        if area > threshold_area:
            relabeled_mask += (labeled_mask == label) * new_label
            new_label += 1

    return relabeled_mask, new_label - 1


def match_regions(labeled_true, num_true, labeled_pred, num_pred, hparams):
    regions_areas = calculate_region_areas(labeled_pred, num_pred)
    sorted_pred_labels = np.argsort(regions_areas)[::-1] + 1
    true_labels = np.array(range(1, num_true + 1))

    tp = 0
    fp = 0
    matched_true_labels = set()
    for i in sorted_pred_labels:
        mask_pred = labeled_pred == i

        best_match = -1
        best_score = hparams.min_iou

        for j in true_labels:
            mask_true = labeled_true == j
            match_score = iou_score(mask_true, mask_pred)

            if (j not in matched_true_labels) and (match_score > best_score):
                best_match = j
                best_score = match_score

        if best_match != -1:
            tp += 1
            matched_true_labels.add(best_match)
        else:
            fp += 1

    fn = 0
    for j in true_labels:
        if j not in matched_true_labels:
            fn += 1
    return tp, fp, fn


def save_df(df, name, hparams):
    output_dir = os.path.join(hparams.output_path, hparams.name)
    os.makedirs(output_dir, exist_ok=True)
    if hparams.test_val:
        output_name = hparams.prefix + f'{hparams.fold}_{name}.csv'
    else:
        output_name = hparams.prefix + f'test_{name}.csv'
    df.to_csv(os.path.join(output_dir, hparams.prefix + output_name))


def test(hparams):
    assert os.path.exists(hparams.config_path)
    with open(os.path.join(hparams.config_path, "config.yaml"), "r") as file:
        model_configs = yaml.load(file)
    model_configs.update(vars(hparams))
    hparams = model_configs = argparse.Namespace(**model_configs)


    if hparams.test_val:
        dataset = PneumothoraxDataset(
            root=hparams.train_val_folder,
            train_val_csv_path=hparams.train_val_csv_path,
            fold=hparams.fold,
            train=False,
        )

        threshold_probs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        threshold_areas = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        pred_masks_dir = os.path.join(hparams.masks_path, "val_prediction", hparams.name)
    elif hparams.test_hold_out:
        dataset = PneumothoraxDataset(
            root=hparams.train_val_folder,
            train_val_csv_path=hparams.train_val_csv_path,
            fold='test',
            train=False,
        )
        assert hparams.threshold_prob is not None
        assert hparams.threshold_area is not None
        threshold_probs = [hparams.threshold_prob]
        threshold_areas = [hparams.threshold_area]
        pred_masks_dir = os.path.join(hparams.masks_path, "hold_out_test_prediction", hparams.name)
    else:
        raise ValueError('Specify of --test_val or --test_hold_out')

    image_names = [os.path.basename(path) for path in dataset.masks]
    classification = pd.DataFrame(index=image_names)
    segmentation = pd.DataFrame(index=image_names)
    detection = pd.DataFrame(index=image_names)

    for gt_mask_path in tqdm.tqdm(dataset.masks):
        image_name = os.path.basename(gt_mask_path)
        gt = skimage.io.imread(gt_mask_path) > 0
        labeled_gt, num_gt = label_mask(gt)
        classification.loc[image_name, 'gt'] = gt.any()

        pred = skimage.io.imread(os.path.join(pred_masks_dir, image_name))
        pred = skimage.transform.resize(pred, gt.shape)

        for threshold_prob in threshold_probs:
            binarized_pred = pred > threshold_prob
            labeled_pred, num_pred = label_mask(binarized_pred)
            regions_areas = calculate_region_areas(labeled_pred, num_pred)

            for threshold_area in threshold_areas:
                final_labeled_pred, final_num_pred = delete_small_regions(labeled_pred, num_pred, threshold_area,
                                                                          regions_areas=regions_areas)
                final_pred = final_labeled_pred > 0

                classification.loc[image_name, f'{threshold_prob}, {threshold_area}'] = final_pred.any()
                segmentation.loc[image_name, f'{threshold_prob}, {threshold_area}'] = dice(gt, final_pred)

                tp, fp, fn = match_regions(labeled_gt, num_gt, final_labeled_pred, final_num_pred, hparams)
                detection.loc[image_name, f'tp {threshold_prob}, {threshold_area}'] = tp
                detection.loc[image_name, f'fp {threshold_prob}, {threshold_area}'] = fp
                detection.loc[image_name, f'fn {threshold_prob}, {threshold_area}'] = fn

    save_df(classification, 'classification', hparams)
    save_df(segmentation, 'segmentation', hparams)
    save_df(detection, 'detection', hparams)

    classification_scores = pd.DataFrame(columns=["sensitivity", "specificity", "precision", "f1"])
    segmentation_scores = pd.DataFrame(columns=['dice'])
    detection_scores = pd.DataFrame(columns=["sensitivity", "precision", "f1"])
    y_true = classification['gt'].values.astype(int)

    for threshold_prob in threshold_probs:
        for threshold_area in threshold_areas:
            y_pred = classification[f'{threshold_prob}, {threshold_area}'].values.astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            classification_scores.loc[f'{threshold_prob}, {threshold_area}'] = \
                [sensitivity, specificity, precision, f1]

            dices = segmentation[f'{threshold_prob}, {threshold_area}'].values.astype(float)
            segmentation_scores.loc[f'{threshold_prob}, {threshold_area}'] = np.mean(dices)

            tp = detection[f'tp {threshold_prob}, {threshold_area}'].values.astype(int).sum()
            fp = detection[f'fp {threshold_prob}, {threshold_area}'].values.astype(int).sum()
            fn = detection[f'fn {threshold_prob}, {threshold_area}'].values.astype(int).sum()
            precision = tp / (tp + fp)
            sensitivity = tp / (tp + fn)
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            detection_scores.loc[f'{threshold_prob}, {threshold_area}'] = [sensitivity, precision, f1]

    classification_scores = classification_scores.sort_values(by=['f1'], ascending=False)
    segmentation_scores = segmentation_scores.sort_values(by=['dice'], ascending=False)
    detection_scores = detection_scores.sort_values(by=['f1'], ascending=False)

    save_df(classification_scores, 'classification_scores', hparams)
    save_df(segmentation_scores, 'segmentation_scores', hparams)
    save_df(detection_scores, 'detection_scores', hparams)



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
    parser.add_argument(
        "--test_val", action="store_true", help="Flag to make prediction for validation")
    parser.add_argument(
        "--test_hold_out", action="store_true",
        help="Flag to make local test (hold out from train split of competition)")
    parser.add_argument(
        "--min_iou", type=float, default=0.5)
    parser.add_argument(
        "--threshold_prob", type=float, default=None, help="")
    parser.add_argument(
        "--threshold_area", type=int, default=None, help="")
    parser.add_argument(
        '--prefix', type=str, default='', help='Prefix for output names')

    hparams = parser.parse_args()
    print(f"Parameters used for test: {hparams}")
    start_time = time.time()
    test(hparams)
    print(f"Finished test. Took: {(time.time() - start_time) / 60:.02f}m")