from glob import glob
import numpy as np
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.funcional as F
from transforms import get_augs
from net import get_net


DATA_ROOT_PATH = "../input"
CHECKPOINT_PATH = "../input/alaska2-public-baseline/best-checkpoint-033epoch.bin"  # change this when you run


class DatasetSubmissionRetriever:
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


def run():
    _, valid_augs = get_augs()
    test_dataset = DatasetSubmissionRetriever(
        image_names=np.array(
            [
                path.split("/")[-1]
                for path in glob("../input/alaska2-image-steganalysis/Test/*.jpg")
            ]
        ),
        transforms=valid_augs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    checkpoint = torch.load(CHECKPOINT_PATH)
    net = get_net()
    net.load_state_dict(checkpoint["model_state_dict"])

    result = {"Id": [], "Label": []}
    for step, (image_names, images) in enumerate(test_loader):
        print(step, end="\r")

        y_pred = net(images.cuda())
        y_pred = (
            1 - F.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
        )  # first column corresponds to 'proba of no hidden code'

        result["Id"].extend(image_names)
        result["Label"].extend(y_pred)

    submission = pd.DataFrame(result)
    submission.to_csv("../submission.csv", index=False)
