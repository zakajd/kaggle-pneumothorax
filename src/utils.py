import functools

import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_tools as pt


MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "segm_fpn": pt.segmentation_models.SegmentationFPN,
    "segm_bifpn": pt.segmentation_models.SegmentationBiFPN,
}


# All this losses expect raw logits as inputs. 
LOSS_FROM_NAME = {
    "bce": pt.losses.CrossEntropyLoss(mode="binary"),
    "wbce": pt.losses.CrossEntropyLoss(mode="binary", weight=[5]),
    "dice": pt.losses.DiceLoss(mode="binary"),
    "jaccard": pt.losses.JaccardLoss(mode="binary", ),
    # "log_jaccard": pt.losses.JaccardLoss(mode="binary", log_loss=True),
    # "hinge": pt.losses.BinaryHinge(),
    # "whinge": pt.losses.BinaryHinge(pos_weight=3),
    "focal": pt.losses.FocalLoss(mode="binary"),
    # "reduced_focal": pt.losses.BinaryFocalLoss(reduced=True),
    "mse": pt.losses.MSELoss(),
    "mae": pt.losses.L1Loss(),
}


def criterion_from_list(criterions):
    """Turn a list of criterias and weights into a weighted criteria.
    Args:
        criterions (list): Something like `[bce, 0.5, dice, 0.5]` to construct loss
    """
    losses = [LOSS_FROM_NAME[loss] for loss in criterions[::2]]
    losses = [loss * float(weight) for loss, weight in zip(losses, criterions[1::2])]
    return functools.reduce(lambda x, y: x + y, losses)


class ToCudaLoader:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, (tuple, list)):
                yield [i.cuda(non_blocking=True) for i in batch]
            else:
                yield batch.cuda(non_blocking=True)

    def __len__(self):
        return len(self.loader)


def apply_thresholds(
    mask,
    min_total_area=2048,
    min_component_area=512,
    top_threshold=0.7, 
    bottom_threshold=0.3):
    """Convert probabilities into binary mask.
    1. Filter by top threshold 
    2. If remaining image is bigger than min_total_area
    3. Apply bottom_threshold
    4. Filter by min_component_area.
    Args:
        mask (np.array): Scores. Shape = (1024, 1024)
        min_total_area (int): Minimal area of a mask
        min_component_area (int): Minimal area of COMPONENT
        top_threshold (float): Used to remove noise
        bottom_threshold (float): Actual threshold
    """
    # Sanity checks
    assert type(mask) == np.ndarray
    assert mask.shape == (1024, 1024)
    assert (0.0 <= mask.min()) & (mask.max() <= 1.0)
    
    empty_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    top_binarized = (mask >= top_threshold).astype(np.uint8)
    if top_binarized.sum() < min_total_area:
        return empty_mask
    
    bottom_binarized = (mask >= bottom_threshold).astype(np.uint8)
    # Find all connected components
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(bottom_binarized, connectivity=8)
    # Remove background
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    
    binary_mask = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_component_area
    for i in range(nb_components):
        if sizes[i] >= min_component_area:
            binary_mask[output == i + 1] = 255
    return binary_mask

