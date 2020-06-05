import os
from datetime import datetime
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from pytorch_tools.optim import optimizer_from_name
import pytorch_tools as pt

from src.data.datasets import get_ssl_dataloaders
from src.utils import MODEL_FROM_NAME, criterion_from_list
from src.ssl.torch_augm import FlipAug, NoiseAug
from src.ssl.ssl_functions import ssl_pseudolabeling, ssl_consistency, ssl_cowmix

get_fn_by_name = {
    'ssl_pseudolabeling': ssl_pseudolabeling,
    'ssl_cowmix': ssl_cowmix,
    'ssl_consistency': ssl_consistency,
    'none': None
}

def epoch_iterator(loader, ssl_loader, ssl_function, train_mode, model, optimizer, criterion, metrics, device, epoch,
                   epoch_max, augm_classes, criterion_ssl):
    if train_mode == 'train':
        model.train()
    else:
        model.eval()
    loss_vals = []
    metrics_vals = []
    ssl_loader_iterator = iter(ssl_loader)
    for idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        pred = model(imgs)

        loss = criterion(pred, masks)
        p_ssl = ((epoch_max - epoch) / epoch_max) / 2 + 0.5
        if train_mode == 'train' and np.random.random() > p_ssl and ssl_function is not None:
            loss_ssl = ssl_function(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl)
            loss = loss + 2*loss_ssl

        if train_mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics_val = metrics(pred, masks)

        loss_vals.append(loss.item())
        metrics_vals.append(metrics_val.item())

    return np.mean(loss_vals), np.mean(metrics_vals)

def setup_experiment(title, logdir="./tb", snapdir="./snapshots"):
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=os.path.join(logdir, experiment_name))
    snapshots_folder = os.path.join(snapdir, experiment_name)
    if not os.path.exists(snapshots_folder):
        os.mkdir(snapshots_folder)
    best_model_path = f"{os.path.join(snapshots_folder, title)}.best.pth"
    return writer, snapshots_folder, best_model_path


class SSLTrainer:

    def __init__(self, hparams):
        self.hparams = hparams
        self.writer, self.snapshots_folder, self.best_model_path = setup_experiment(hparams.name, hparams.logdir,
                                                                                    hparams.snapdir)
        yaml.dump(vars(hparams), open(os.path.join(self.snapshots_folder, 'config.yaml'), 'w'))
        self.model = MODEL_FROM_NAME[hparams.segm_arch](hparams.backbone, **hparams.model_params).to(hparams.device)
        self.optimizer = optimizer_from_name(hparams.optim)(
            self.model.parameters(),  # Get LR from phases later
            lr=hparams.lr,
            weight_decay=hparams.weight_decay
        )

        scheduler_class = globals()[hparams.scheduler]
        hparams.scheduler_params['optimizer'] = self.optimizer
        self.scheduler = scheduler_class(**hparams.scheduler_params)

        self.criterion = criterion_from_list(hparams.criterion).to(hparams.device)
        self.metrics_dice = pt.metrics.DiceScore(mode="binary").to(hparams.device)
        self.ssl_function = get_fn_by_name[hparams.ssl_function]
        self.ssl_augm_classes = [FlipAug((2,)), FlipAug((3,)), FlipAug((2, 3,)), NoiseAug(0.1)]
        self.ssl_criterion = torch.nn.MSELoss()

    def fit_ssl(self):
        best_dice = 0
        for ep in range(self.hparams.epochs):
            pos_weight = self.hparams.pos_weight_start - ep * (
                    self.hparams.pos_weight_start - self.hparams.pos_weight_end) / self.hparams.epochs
            train_loader, val_loader, ssl_loader = get_ssl_dataloaders(self.hparams.train_val_folder,
                                                                       self.hparams.train_val_csv_path,
                                                                       self.hparams.train_size,
                                                                       self.hparams.val_size, str(self.hparams.fold),
                                                                       self.hparams.ssl_path, self.hparams.augmentation,
                                                                       pos_weight, self.hparams.batch_size,
                                                                       self.hparams.workers)
            train_loss, train_dice = epoch_iterator(train_loader, ssl_loader, self.ssl_function, 'train', self.model,
                                                    self.optimizer,
                                                    self.criterion, self.metrics_dice, self.hparams.device, ep,
                                                    self.hparams.epochs, self.ssl_augm_classes, self.ssl_criterion)
            with torch.set_grad_enabled(False):
                val_loss, val_dice = epoch_iterator(val_loader, ssl_loader, self.ssl_function, 'val', self.model,
                                                    self.optimizer, self.criterion, self.metrics_dice,
                                                    self.hparams.device, ep, self.hparams.epochs,
                                                    self.ssl_augm_classes, self.ssl_criterion)

            if self.writer is not None:
                self.writer.add_scalar(f"loss/train", train_loss, ep)
                self.writer.add_scalar(f"dice/train", train_dice, ep)
                self.writer.add_scalar(f"loss/val", val_loss, ep)
                self.writer.add_scalar(f"dice/val", val_dice, ep)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(self.model.state_dict(), self.best_model_path)
            self.scheduler.step()
            print('Epoch', ep)
            print(f"Train_loss:{train_loss} Train_dice:{train_dice}")
            print(f"Val_loss:{val_loss} Val_dice:{val_dice}")