import numpy as np
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

# From - https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/7
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, device='cpu'):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        kernel = kernel.to(device)
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.weight.shape[-1] // 2)

def sample_from_loader(ssl_loader, ssl_loader_iterator):
    try:
        img_u = ssl_loader_iterator.next()
    except StopIteration:
        ssl_loader_iterator = iter(ssl_loader)
        img_u = ssl_loader_iterator.next()
    return img_u

def ssl_pseudolabeling(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl):
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
    img_u = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u = img_u.to(device)
    aug_function = np.random.choice(augm_classes)
    pred = model(img_u)
    pred_aug = aug_function.reverse(model(aug_function.direct(img_u)))
    loss = criterion_ssl(torch.sigmoid(pred), torch.sigmoid(pred_aug))

    return loss

def ssl_cowmix(model, ssl_loader, ssl_loader_iterator, imgs, masks, device, augm_classes, criterion_ssl):
    p = 0.5
    img_u1 = sample_from_loader(ssl_loader, ssl_loader_iterator)
    img_u1 = img_u1.to(device)

    img_u2 = sample_from_loader(ssl_loader)
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




