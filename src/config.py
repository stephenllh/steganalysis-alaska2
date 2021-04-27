import torch


class TrainGlobalConfig:
    csv_file = "train_75.csv"  # can change to train_90.csv or train_95.csv
    fold_number = 0
    num_workers = 4
    batch_size = 16
    n_epochs = 5
    lr = 2e-4

    verbose = True
    verbose_step = 1

    step_scheduler = True  # do scheduler.step after optimizer.step
    valid_scheduler = False  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=None,
        pct_start=0.1,
        anneal_strategy="cos",
        cycle_momentum=True,
        div_factor=10.0,
    )
