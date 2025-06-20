import numpy as np
from torch import optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    # https://pytorch-lightning.readthedocs.io/en/latest/\
    # notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def __init__(self, optimizer, warmup, max_iters, verbose):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        # if(self.last_epoch == 0):
        #     return self.base_lrs
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
