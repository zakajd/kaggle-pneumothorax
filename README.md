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
