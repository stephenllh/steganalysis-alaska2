from torch import nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.training:
            logits = logits.float()
            targets = targets.float()

            log_probs = F.log_softmax(logits, dim=-1)

            nll_loss = (-log_probs * targets).sum(-1)
            smooth_loss = -log_probs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()

        else:
            return F.cross_entropy(logits, targets)
