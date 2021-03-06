import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt

## Default ImageNet mean and std
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def get_aug(aug_type="val", size=512):
    """Return augmentations by type
    Args:
        aug_type (str): one of `val`, `test`, `light`, `medium`
        size (int): final size of the crop
    """

    NORM_TO_TENSOR = albu.Compose([
        albu.Normalize(mean=MEAN, std=STD), 
        albu.pytorch.ToTensorV2(),
    ])

    CROP_AUG = albu.Resize(size, size, always_apply=True)

    VAL_AUG = albu.Compose([
        albu.Resize(size, size),
        NORM_TO_TENSOR])

    TEST_AUG = albu.Compose([
        albu.Resize(size, size),
        NORM_TO_TENSOR,
    ])

    LIGHT_AUG = albu.Compose([CROP_AUG, albu.HorizontalFlip(), NORM_TO_TENSOR])

    MEDIUM_AUG = albu.Compose([
        CROP_AUG,
        albu.HorizontalFlip(),
        # Spatial-preserving augmentations:
        albu.OneOf([
            albu.RandomContrast(),
            albu.RandomGamma(),
            albu.RandomBrightness(),
            ], p=0.3),
        albu.OneOf([
            albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(),
            albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        albu.ShiftScaleRotate(rotate_limit=15),
        NORM_TO_TENSOR,
    ])
    

    types = {
        "val" : VAL_AUG,
        "test" : TEST_AUG,
        "light" : LIGHT_AUG,
        "medium" : MEDIUM_AUG,
    }

    return types[aug_type]