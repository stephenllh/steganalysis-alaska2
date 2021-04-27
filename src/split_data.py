import glob
import random
import pandas as pd
from sklearn.model_selection import GroupKFold


def group_k_fold():
    dataset = []

    for label, kind in enumerate(["Cover", "JMiPOD", "JUNIWARD", "UERD"]):
        for path in glob("../input/alaska2-image-steganalysis/Cover/*.jpg"):
            dataset.append(
                {"kind": kind, "image_name": path.split("/")[-1], "label": label}
            )

    random.shuffle(dataset)
    dataset = pd.DataFrame(dataset)

    gkf = GroupKFold(n_splits=5)

    dataset.loc[:, "fold"] = 0
    for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
    ):
        dataset.loc[dataset.iloc[val_index].index, "fold"] = fold_number
