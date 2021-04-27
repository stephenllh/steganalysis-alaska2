import numpy as np
import cv2
import torch


DATA_ROOT_PATH = "../input/alaska2-image-steganalysis"


class Dataset:
    def __init__(self, df, num_classes=4, transforms=None):
        super().__init__()
        self.df = df
        self.num_classes = num_classes
        self.transforms = transforms

    def __getitem__(self, index):
        filename = (self.df["filename"].values)[index]
        image = cv2.imread(f"{DATA_ROOT_PATH}/{filename}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        label_idx = (self.df["label"].values)[index]
        target = self.onehot(self.num_classes, label_idx)
        return image, target

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return list(self.df["label"].values)

    def onehot(self, num_classes, target):
        vec = torch.zeros(num_classes, dtype=torch.float32)
        vec[target] = 1.0
        return vec


class TestDataset:
    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = cv2.imread(f"{DATA_ROOT_PATH}/Test/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        return image_name, image

    def __len__(self):
        return self.image_names.shape[0]
