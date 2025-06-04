from typing import Optional

import math
import numpy as np
import torch
from torch import optim


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    # https://pytorch-lightning.readthedocs.io/en/latest/\
    # notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            t_initial: int,
            lr_min: float = 0.,
            cycle_mul: float = 1.,
            cycle_decay: float = 1.,
            cycle_limit: int = 1,
            warmup_t=0,
            warmup_lr_init=0,
            warmup_prefix=False,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            k_decay=1.0,
            initialize=True,

    ):
        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and cycle_mul == 1 and cycle_decay == 1:
            print(
                "Cosine annealing scheduler will have no effect on the learning "
                "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay

        self.optimizer = optimizer
        self.t_in_epochs = t_in_epochs
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = 'normal'
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.param_group_field = "lr"
        self._initial_param_group_field = f"initial_{self.param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if self.param_group_field not in group:
                    raise KeyError(f"{self.param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[self.param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.update_groups(self.base_values)
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            self.update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        super().__init__(optimizer)

    def _get_lr(self, t):
        t = self.last_epoch
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul))
                t_i = self.cycle_mul ** i * self.t_initial
                t_curr = t - (1 - self.cycle_mul ** i) / (1 - self.cycle_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay ** i
            lr_max_values = [v * gamma for v in self.base_values]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min + 0.5 * (lr_max - self.lr_min) * (1 + math.cos(math.pi * t_curr ** k / t_i ** k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def _get_values(self, t: int, on_epoch: bool = True) -> Optional[float]:
        proceed = (on_epoch and self.t_in_epochs) or (not on_epoch and not self.t_in_epochs)
        if not proceed:
            return None
        return self._get_lr(t)

    def get_lr(self):
        epoch = self.last_epoch
        values = self._get_values(epoch, on_epoch=True)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)
        return values

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if 'lr_scale' in param_group:
                param_group[self.param_group_field] = value * param_group['lr_scale']
            else:
                param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self._is_apply_noise(t):
            noise = self._calculate_noise(t)
            lrs = [v + v * noise for v in lrs]
        return lrs

    def _is_apply_noise(self, t) -> bool:
        """Return True if scheduler in noise range."""
        apply_noise = False
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
        return apply_noise

    def _calculate_noise(self, t) -> float:
        g = torch.Generator()
        g.manual_seed(self.noise_seed + t)
        if self.noise_type == 'normal':
            while True:
                # resample if noise out of percent limit, brute force but shouldn't spin much
                noise = torch.randn(1, generator=g).item()
                if abs(noise) < self.noise_pct:
                    return noise
        else:
            noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
        return noise

