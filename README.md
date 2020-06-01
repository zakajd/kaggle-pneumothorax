# kaggle-pneumothorax
Solution of Kaggle Pnumothorax challenge

## Installation
1. Clon the repo
2. Install all the requirements `pip install --user -r requirements.txt`
To reproduce the solution, run following commands:

```
make load
make preprocess
```

## Code organization

    ├── README.md             <- Top-level README.
    ├── Makefile        <-  Used to make everything in this project
    │
    ├── confings    <- Parameters for traing 
    │   ├── default.yaml         <- Default parameters
    │   ├── unet.yaml          <- Over-write some default values for model training
    │
    ├── data       <- All data must be here
    |
    ├── docker        <- Container for easy reprodusable solution (NOT working yet)
    ├── logs        <- TensorBoard logging to monitor training
    ├── models        <- Pretrained models saved as `*.ckpt`
    ├── notebooks        <- Jupyter Notebooks
    │   ├── Testing.ipynb   <- Develompent related stuff                  
    │   ├── Demo.ipynb  <- Demonstation of the results
    ├── src        <- Code


## Usage

### inference.py

Script takes a model and saves predicted probabilities for validation split (--predict_val) or 
for test split (--predict_test, not implemented yet)

Expected that 

    1. `config.yaml` and `model.chpn` lies in the same directory, 
    2. `config.yaml` contains variables:
    
        * name
        * everything for val dataloader:  
            * root,
            * fold,
            * val_size,
            * batch_size,
            * workers
            
```
usage: inference.py [-h] [--config_path CONFIG_PATH] [--output_path OUTPUT_PATH] [--predict_val] [--predict_test]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to folder with model config and checkpoint
  --output_path OUTPUT_PATH
                        Path to save masks
  --predict_val         Flag to make prediction for validation
  --predict_test        Flag to make prediction for test

```       

#### Example

Data structure:

    
    ├── data
    │   ├── logs
    |   |   ├── unet_resnet
    |   |   |   ├── config.yaml
    |   |   |   ├── model.chpn
    │   ├── processed

Run:

```bash
    python inference.py --config_path data/logs/unet_resnet --output_path data/processed/  --predict_val  
```        

Result:

    ├── data
    │   ├── logs
    |   |   ├── unet_resnet
    |   |   |   ├── config.yaml
    |   |   |   ├── model.chpn
    │   ├── processed
    |   |   ├── unet_resnet
    |   |   |   ├── img1.png
    |   |   |   ├── ...
    |   |   |   ├── imgn.png
    
    
### test.py

Script takes a ground truth masks and predicted probabilities, and computes different classification, segmetation and detection metrics.

```
usage: test.py [-h] [--config_path CONFIG_PATH] [--masks_path MASKS_PATH] [--output_path OUTPUT_PATH] [--min_iou MIN_IOU]
               [--prefix PREFIX]

optional arguments:
  -h, --help            show this help message and exit
  --config_path CONFIG_PATH
                        Path to folder with model config and checkpoint
  --masks_path MASKS_PATH
                        Root path with masks
  --output_path OUTPUT_PATH
                        Path to save scores
  --min_iou MIN_IOU
  --prefix PREFIX       Prefix for output names

```       

#### Example

Data structure:

    
    ├── data
    │   ├── logs
    |   |   ├── unet_resnet
    |   |   |   ├── config.yaml
    │   ├── processed
    |   |   ├── unet_resnet
    |   |   |   ├── img1.png
    |   |   |   ├── ...
    |   |   |   ├── imgn.png
    │   ├── scores
Run:

```bash
    python test.py --config_path data/logs/unet_resnet --masks_path data/processed/  --output_path data/scores  
```        

Result:

    ├── data
    │   ├── logs
    |   |   ├── unet_resnet
    |   |   |   ├── config.yaml
    |   |   |   ├── model.chpn
    │   ├── processed
    |   |   ├── unet_resnet
    |   |   |   ├── img1.png
    |   |   |   ├── ...
    |   |   |   ├── imgn.png
    │   ├── scores
    |   |   ├── unet_resnet
    |   |   |   ├── {fold}_segmentation_scores.csv
    |   |   |   ├── {fold}_detection_scores.csv
    |   |   |   ├── {fold}_classificatio_scores.csv
    |   |   |   ├── ...


### merge_scores_over_folds.py

Script takes .cvs for different folds and merges it in one file computing mean and std over folds.


```
usage: merge_scores_over_folds.py [-h] [--prefix PREFIX] root

positional arguments:
  root

optional arguments:
  -h, --help       show this help message and exit
  --prefix PREFIX  Prefix for output names
```   


#### Example

Data structure:

    
    ├── data
    │   ...
    │   ├── scores
    |   |   ├── unet_resnet
    |   |   |   ├── {fold}_segmentation_scores.csv
    |   |   |   ├── {fold}_detection_scores.csv
    |   |   |   ├── {fold}_classificatio_scores.csv
    |   |   |   ├── ...
Run:

```bash
    python merge_scores_over_folds.py data/scores/unet_resnet
```        

Result:

    ├── data
    │   ...
    │   ├── scores
    |   |   ├── unet_resnet
    |   |   |   ├── {fold}_segmentation_scores.csv
    |   |   |   ├── {fold}_detection_scores.csv
    |   |   |   ├── {fold}_classificatio_scores.csv
    |   |   |   ├── ...
    |   |   |   ├── segmentation_scores.csv
    |   |   |   ├── classificatio_scores.csv
    |   |   |   ├── detection_scores.csv
    