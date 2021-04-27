import pandas as pd
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from dataset import Dataset
from transforms import get_augs
from net import get_net
from config import TrainGlobalConfig
from learner import Learner
from util import seed_everything


def run():
    SEED = 42
    seed_everything(SEED)
    csv_file = TrainGlobalConfig.csv_file
    df = pd.read_csv(f"../input/metadata/{csv_file}")

    train_augs, valid_augs = get_augs()
    train_dataset = Dataset(
        df=df[df["fold"] != TrainGlobalConfig.fold_number],
        transforms=train_augs,
    )

    valid_dataset = Dataset(
        df=df[df["fold"] == TrainGlobalConfig.fold_number],
        transforms=valid_augs,
    )

    train_loader = DataLoader(
        train_dataset,
        sampler=BalanceClassSampler(
            labels=train_dataset.get_labels(), mode="downsampling"
        ),
        batch_size=TrainGlobalConfig.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(valid_dataset),
        pin_memory=False,
    )
    TrainGlobalConfig.scheduler_params["steps_per_epoch"] = (
        len(train_dataset) // TrainGlobalConfig.batch_size
    )

    net = get_net().cuda()
    learner = Learner(model=net, config=TrainGlobalConfig)
    learner.fit(train_loader, valid_loader)
