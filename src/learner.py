from glob import glob
import os
from datetime import datetime
import time
import torch
from loss import LabelSmoothing
from metric import AverageMeter, RocAucMeter
import warnings

warnings.filterwarnings("ignore")


class Learner:
    def __init__(self, model, config, base_dir="./"):
        self.model = model.cuda()
        self.config = config

        self.base_dir = base_dir
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_loss = 1e5

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().cuda()
        self.log("Learner prepared.")

    def fit(self, train_loader, valid_loader):
        for epoch in range(self.config.n_epochs):
            if self.config.verbose:
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\n")

            # Training
            t = time.time()
            train_loss, auc_scores = self.train(train_loader)

            self.log(
                f"[RESULT]: Train. Epoch: {epoch}, train_loss: {train_loss.avg:.5f}, auc_score: {auc_scores.avg:.5f}, time: {(time.time() - t):.5f}"
            )
            self.save(f"{self.base_dir}/last-checkpoint.bin")

            # Validation
            t = time.time()
            valid_loss, auc_scores = self.validation(valid_loader)

            self.log(
                f"[RESULT]: Val. Epoch: {epoch}, valid_loss: {valid_loss.avg:.5f}, auc_score: {auc_scores.avg:.5f}, time: {(time.time() - t):.5f}"
            )
            if valid_loss.avg < self.best_loss:
                self.best_loss = valid_loss.avg
                self.save(
                    f"{self.base_dir}/fold{self.config.fold_number}-best-checkpoint-{str(epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(
                    glob(
                        f"{self.base_dir}/fold{self.config.fold_number}-best-checkpoint-*epoch.bin"
                    )
                )[:-3]:
                    os.remove(path)

            if self.config.valid_scheduler:
                self.scheduler.step(metrics=valid_loss.avg)

    def train(self, train_loader):

        self.model.train()

        train_loss = AverageMeter()
        auc_scores = RocAucMeter()

        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Train step {step}/{len(train_loader)}, Learning rate = {1e6*lr:.6f}e-6, "
                        + f"Train loss: {train_loss.avg:.5f}, AUC score: {auc_scores.avg:.5f}, "
                        + f"Time: {(time.time() - t):.5f}",
                        end="\r",
                    )

            images = images.cuda().float()
            targets = targets.cuda().float()

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, targets)
            loss.backward()
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

            auc_scores.update(preds, targets)
            batch_size = images.shape[0]
            train_loss.update(loss.detach().item(), batch_size)

        return train_loss, auc_scores

    def validation(self, valid_loader):

        self.model.eval()

        valid_loss = AverageMeter()
        auc_scores = RocAucMeter()

        t = time.time()
        for step, (images, targets) in enumerate(valid_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f"Validation step {step}/{len(valid_loader)}, "
                        + f"Valid loss: {valid_loss.avg:.5f}, AUC score: {auc_scores.avg:.5f}, "
                        + f"Time: {(time.time() - t):.5f}",
                        end="\r",
                    )

            with torch.no_grad():
                images = images.cuda().float()
                targets = targets.cuda().float()
                preds = self.model(images)
                loss = self.criterion(preds, targets)

                auc_scores.update(preds, targets)
                batch_size = images.shape[0]
                valid_loss.update(loss.detach().item(), batch_size)

        return valid_loss, auc_scores

    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_loss": self.best_loss,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_loss = checkpoint["best_loss"]

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+") as logger:
            logger.write(f"{message}\n")
