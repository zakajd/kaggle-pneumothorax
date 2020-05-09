import os
import yaml
import time

import apex
import torch
from loguru import logger
import pytorch_tools as pt
import pytorch_tools.fit_wrapper.callbacks as pt_clb 
from pytorch_tools.optim import optimizer_from_name
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClb

from src.models.arg_parser import parse_args
from src.data.datasets import get_dataloaders
from src.utils import MODEL_FROM_NAME, criterion_from_list
from src.callbacks import PredictViewer

def main():
    # Setup logger
    config = {
    "handlers": [ 
        {"sink": sys.stdout, "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        {"sink": f"{hparams.outdir}/logs.txt", "format": "{time:[MM-DD HH:mm:ss]} - {message}"},
        ],
    }
    logger.configure(**config)

    # Get config for this run
    hparams = parse_args()
    logger.info(f"Parameters used for training: {hparams}")

    # Fix seeds for reprodusability
    pt.utils.misc.set_random_seed(hparams.seed) 

    ## Save config and Git diff (don't know how to do it without subprocess)
    os.makedirs(hparams.outdir, exist_ok=True)
    yaml.dump(vars(hparams), open(hparams.outdir + '/config.yaml', 'w'))
    kwargs = {"universal_newlines": True, "stdout": subprocess.PIPE}
    with open(hparams.outdir + '/commit_hash.txt', 'w') as f:
        f.write(subprocess.run(["git", "rev-parse", "--short", "HEAD"], **kwargs).stdout)
    with open(hparams.outdir + '/diff.txt', 'w') as f:
        f.write(subprocess.run(["git", "diff"], **kwargs).stdout)

    ## Get dataloaders
    train_loader, val_loader = get_dataloaders(
        root=hparams.root, 
        augmentation=hparams.augmentation,
        fold=hparams.fold,
        pos_weight=hparams.pos_weight,
        batch_size=hparams.batch_size,
        size=hparams.size, 
        val_size=hparams.val_size,
        workers=hparams.workers
    )

    # Get model and optimizer
    model = MODEL_FROM_NAME[hparams.segm_arch](hparams.backbone, **hparams.model_params).cuda()
    optimizer = optimizer_from_name(hparams.optim)(
        model.parameters(), # Get LR from phases later
        weight_decay=hparams.weight_decay
    )

    # Convert all Conv2D -> WS_Conv2d if needed
    if hparams.ws:
        model = pt.modules.weight_standartization.conv_to_ws_conv(model).cuda()

    # Load weights if needed
    if hparams.resume:
        checkpoint = torch.load(hparams.resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    num_params = pt.utils.misc.count_parameters(model)[0]
    logger.info(f"Model size: {num_params / 1e6:.02f}M")  

    ## Use AMP
    model, optimizer = apex.amp.initialize(
        model, optimizer, opt_level=hparams.opt_level, verbosity=0, loss_scale=1024
    )

    # Get loss
    loss = criterion_from_list(hparams.criterion).cuda()
    logger.info(f"Loss for this run is: {loss}")

    bce_loss = pt.losses.CrossEntropyLoss(mode="binary").cuda() # Used as a metric
    bce_loss.name = "BCE"

    # Scheduler is an advanced way of planning experiment
    sheduler = pt.fit_wrapper.callbacks.PhasesScheduler(hparams.phases)

    # Init runner 
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt_clb.Timer(),
            pt_clb.ConsoleLogger(),
            pt_clb.FileLogger(hparams.outdir, logger=logger),
            pt_clb.CheckpointSaver(hparams.outdir, save_name="model.chpn"),
            sheduler,
            PredictViewer(hparams.outdir, num_images=4)
        ],
        metrics=[
            bce_loss,
            pt.metrics.JaccardScore(mode="binary").cuda(),
            # ThrJaccardScore(thr=0.5),
        ],
    )

    if hparams.decoder_warmup_epochs > 0:
        # Freeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = False

        runner.fit(
            train_loader,
            val_loader=val_loader,

            epochs=hparams.decoder_warmup_epochs,
            steps_per_epoch=10 if hparams.debug else None,
            val_steps=10 if hparams.debug else None,
            # val_steps=50 if hparams.debug else None,
        )

        # Unfreeze all
        for p in model.parameters():
            p.requires_grad = True

        # Reinit again to avoid NaN's in loss
        optimizer = optimizer_from_name(hparams.optim)(
            model.parameters(),
            weight_decay=hparams.weight_decay
        )
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level=hparams.opt_level, verbosity=0, loss_scale=2048
        )
        runner.state.model = model
        runner.state.optimizer = optimizer

    # Train both encoder and decoder
    runner.fit(
        train_loader,
        val_loader=val_loader,
        start_epoch=hparams.decoder_warmup_epochs,
        epochs=sheduler.tot_epochs,
        steps_per_epoch=10 if hparams.debug else None,
        val_steps=10 if hparams.debug else None,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")