import sys

import pytorch_lightning as pl
from tqdm import tqdm


class Bar(pl.callbacks.TQDMProgressBar):
    def __init__(self):
        super().__init__()

    def init_validation_tqdm(self):
        # has_main_bar = self.trainer.state.fn != "validate"
        print(" ")
        bar = tqdm(
            desc=self.validation_description,
            position=0,
            disable=self.is_disabled,
            leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            ncols=100,
            mininterval=0.05,
            colour='red',
        )
        return bar

    def init_train_tqdm(self):
        bar = tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            # dynamic_ncols=True,
            file=sys.stdout,
            ncols=100,
            mininterval=0.05,
            colour='green',
        )
        return bar