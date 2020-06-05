import numpy as np
import torch
from src.ssl.utils import GaussianSmoothing

def sample_from_loader(ssl_loader, ssl_loader_iterator):
    # Return a pytorch tensor of unlabeled data
    try:
        img_u = ssl_loader_iterator.next()
    except StopIteration:
        ssl_loader_iterator = iter(ssl_loader)
        img_u = ssl_loader_iterator.next()
    return img_u

def ssl_pseudolabeling(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl):
    # Return semi-supervised loss
    # Calculates pseudolabels and a corresponding loss
    t = 0.25
    img_u = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u = img_u.to(device)
    augm_outputs = torch.zeros((len(augm_classes) + 1,) + img_u[:,:1].shape, device=device, dtype=torch.float32)
    augm_outputs[0] = model(img_u).detach()

    for idx, augm_class in enumerate(augm_classes):
        augm_outputs[idx+1] = augm_class.reverse(model(augm_class.direct(img_u))).detach()

    mean_preds = torch.sigmoid(augm_outputs).mean(dim=0)
    mean_preds = mean_preds**(1/t) / ((1 - mean_preds)**(1/t) + mean_preds**(1/t))

    preds = torch.sigmoid(model(img_u))
    loss = criterion_ssl(preds, mean_preds)
    return loss

def ssl_consistency(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl):
    # Return semi-supervised loss
    # Calculates consistency loss using a random reversable transform
    img_u = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u = img_u.to(device)
    aug_function = np.random.choice(augm_classes)
    pred = model(img_u)
    pred_aug = aug_function.reverse(model(aug_function.direct(img_u)))
    loss = criterion_ssl(torch.sigmoid(pred), torch.sigmoid(pred_aug))

    return loss

def ssl_cowmix(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl):
    # Return semi-supervised loss
    # Calculate cow-mix loss from https://arxiv.org/abs/1906.01916
    p = 0.5
    img_u1 = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u1 = img_u1.to(device)

    img_u2 = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u2 = img_u2.to(device)

    noise = torch.randn_like(img_u2[:,:1], device=device)
    g_sm = GaussianSmoothing(1, 65, 8, device=device)
    noise = g_sm(noise)
    thr = torch.erf((2*p - 1)*np.sqrt(2)*noise.std(dim=(1,2,3)) + noise.mean(dim=(1,2,3)))

    mask_segm1 = (noise >= thr.view(-1, 1, 1, 1)).type(torch.float32)
    mask_segm2 = (noise < thr.view(-1, 1, 1, 1)).type(torch.float32)

    mask_img1 = mask_segm1.repeat_interleave(3,1)
    mask_img2 = mask_segm2.repeat_interleave(3,1)

    pred_u1 = torch.sigmoid(model(img_u1))
    pred_u2 = torch.sigmoid(model(img_u2))

    mixed_img = img_u1*mask_img1 + img_u2*mask_img2
    pred_mixed = torch.sigmoid(model(mixed_img))

    loss = criterion_ssl(pred_mixed, pred_u1*mask_segm1 + pred_u2*mask_segm2)

    return loss




