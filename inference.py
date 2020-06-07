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
import albumentations as albu

# Local imports
from src.arg_parser import get_parser
from src.data.dataset import get_test_dataloader
from src.utils import MODEL_FROM_NAME
from src.callbacks import ThrJaccardScore


@torch.no_grad()
def predict_from_loader(model, loader, file_names, output_path):
    """Predict, threshold, save
    Args:
        model
        loader
        file_names (list)
        output_path (str)
    """
    # center_crop = albu.CenterCrop(900, 900)

    idx = 0
    for batch in tqdm(loader):
        # Take images if (images, masks) returned
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # B x 4 x H x W -> B x H x W 
        prediction = model(batch).sigmoid().cpu().squeeze()

        for p in prediction:
            binary_p = (p > 0.5).long().numpy() * 255
            mask = center_crop(image=binary_p)["image"]
            cv2.imwrite(f"{output_path}/{file_names[idx]}.png", mask)
            idx += 1


@torch.no_grad()          
def submit_from_loader(model, loader, file_names, output_path, save=True):
    """Predict, threshold, center crop (to 900 x 900 size), save
    Args:
        model
        loader
        file_names (list)
        output_path (str)
    """
    center_crop = albu.CenterCrop(900, 900)

    idx = 0
    dfs = []
    for batch in tqdm(loader):
        # Take images if (images, masks) returned
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # B x 4 x H x W -> B x H x W 
        prediction = model(batch).sigmoid().cpu().squeeze()

        for p in prediction:
            binary_p = (p > 0.5).long().numpy() * 255
            mask = center_crop(image=binary_p)["image"].astype(np.uint8)
            # print(mask.dtype, mask.min(), mask.max())
            if save:
                cv2.imwrite(f"{output_path}/{file_names[idx]}.png", mask)
            poly_gdf = mask_to_polygons(mask)
            df = pd.DataFrame({
                'ImageId': file_names[idx],
                'PolygonWKT_Pix': poly_gdf['geometry'],
                'Confidence': 1,
            })
            dfs.append(df)
            idx += 1

    solution_df = pd.concat(dfs)
    return solution_df
            

def main(hparams):
    assert os.path.exists(hparams.config_path)
    # Add model parameters 
    with open(hparams.config_path + "/config.yaml", "r") as file:
        model_configs = yaml.load(file)
    vars(hparams).update(model_configs)

    print("Loading model")
    model = MODEL_FROM_NAME[hparams.segm_arch](hparams.backbone, **hparams.model_params)

    # Convert all Conv2D -> WS_Conv2d if needed
    if hparams.ws:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model).cuda()

    checkpoint = torch.load(hparams.path + "/model.chpn")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.cuda().eval()

    if FLAGS.tta:
        model = pt.tta_wrapper.TTA(
            model, segm=True, h_flip=True, h_shift=[10, -10], v_shift=[10, -10], 
            merge="gmean", activation="sigmoid"
        )

    model = apex.amp.initialize(model, verbosity=0)
    print("Model loaded succesfully")

    # Get dataloaders
    test_loader, test_files = get_test_dataloader(root=hparams.root)


    # if hparams.predict_val:
        # runner = pt.fit_wrapper.Runner(
        #     model, 
        #     None, 
        #     pt.losses.JaccardLoss(), 
        #     [
        #         pt.metrics.JaccardScore(),
        #         ThrJaccardScore(thr=0.5),
        #     ],
        # )
        # _, (jacc_score, thr_jacc_score) = runner.evaluate(val_loader)
        # print(f"Validation Jacc Score: {thr_jacc_score:.4f}")
        # output_path = hparams.output_path + "/val_prediction/" + hparams.name
        # print(f"Saving validation masks to {output_path}")

        # # Delete old masks if any
        # shutil.rmtree(output_path, ignore_errors=True)
        # Path(output_path).mkdir(parents=True, exist_ok=False)

        # # Shorten TIFF file name to <timestamp>-<timestamp>-tile-<tile_num>
        # val_names = ["_".join(Path(p).stem.split("_")[-4:])for p in val_files]

        # if hparams.create_solution:
        #     val_pred_df = submit_from_loader(model, val_loader, val_names, output_path, save=True)
        # else:
        #     predict_from_loader(model, val_loader, val_names, output_path)

    if hparams.predict_test:
        output_path = hparams.output_path + "/test_prediction/" + hparams.name
        print(f"Saving test masks to {output_path}")

        # Delete old masks if any
        shutil.rmtree(output_path, ignore_errors=True)
        Path(output_path).mkdir(parents=True, exist_ok=False)

        # Shorten TIFF file name to <timestamp>-<timestamp>-tile-<tile_num>
        # test_names = ["_".join(Path(p).stem.split("_")[-4:])for p in test_files]

        # if hparams.create_solution:
        #     test_pred_df = submit_from_loader(model, test_loader, test_names, output_path, save=True)
        #     test_pred_df.to_csv(f"{hparams.path}/solution.csv", index=False)
        # else:
            predict_from_loader(model, test_loader, test_names, output_path)

    # if hparams.evaluate_val:
    #     true_labels_df = pd.read_csv(hparams.root + "/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv")
    #     true_labels_df = true_labels_df[["ImageId", "PolygonWKT_Pix"]]

    #     val_true_df = true_labels_df[true_labels_df.ImageId.isin(val_names)].reset_index(drop=True)
    #     val_true_df["PolygonWKT_Pix"] = val_true_df["PolygonWKT_Pix"].apply(lambda x: shapely.wkt.loads(x)) # str => Polygon
        
    #     # Get area of each polygon in advance
    #     val_true_df["area"] = val_true_df["PolygonWKT_Pix"].apply(lambda x: x.area)
    #     val_pred_df["area"] = val_pred_df["PolygonWKT_Pix"].apply(lambda x: x.area)
        
    #     f1 = eval_spacenet(val_pred_df, val_true_df, min_area=80)
    #     print(f"F1 score: {f1:.4f}")
    

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
        "--predict_test", action="store_true", help="Flag to make prediction for test")   
    parser.add_argument(
        "--create_solution", action="store_true", help="Flag to transform masks into CSV")  
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode only inferences 5 images") 

    hparams = parser.parse_args()
    print(f"Parameters used for inference: {hparams}")
    start_time = time.time()
    main(hparams)
    print(f"Finished inference. Took: {(time.time() - start_time) / 60:.02f}m")