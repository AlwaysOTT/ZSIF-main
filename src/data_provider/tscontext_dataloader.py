from typing import Dict, Any, Optional

from src.data_provider.my_sampler import SequentialDistributedSampler
from src.datasets import tscontext_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class TSContextDataModule():
    def __init__(
            self,
            dataset: Dict[str, Any],
            batch_size: int = 16,
            num_workers: int = 0,
            pin_memory: bool = False,
    ):

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def data_provider(self, flag, logger, cfg):

        Data = tscontext_dataset.TSContextDataset
        data_set = Data(**self.dataset, mode=flag)
        logger.info(f"{flag}:{len(data_set)}")

        if cfg.multi_gpu:
            if flag == 'train':
                shuffle_flag = False
                drop_last = False
                sampler = DistributedSampler(data_set)
            else:
                shuffle_flag = False
                drop_last = False
                sampler = SequentialDistributedSampler(data_set, batch_size=self.batch_size)
        else:
            if flag == 'train':
                shuffle_flag = True
                drop_last = False
                sampler = None
            else:
                shuffle_flag = False
                drop_last = False
                sampler = None

        data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            sampler=sampler
        )
        return data_set, data_loader
