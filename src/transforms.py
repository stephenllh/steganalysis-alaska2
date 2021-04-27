import albumentations
from albumentations.pytorch.transforms import ToTensorV2


def get_augs():
    train_augs = albumentations.Compose(
        [
            albumentations.Resize(height=512, width=512, p=1.0),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    valid_augs = albumentations.Compose(
        [
            albumentations.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    return train_augs, valid_augs
