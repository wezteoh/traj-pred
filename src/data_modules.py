import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        bsz: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.bsz = bsz
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data = np.load(self.data_dir / "train.npy", allow_pickle=True)
        test_data = np.load(self.data_dir / "test.npy", allow_pickle=True)
        self.dataset = {
            "train": train_data,
            "test": test_data,
        }

    def train_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["train"], shuffle=True)

    def val_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def test_dataloader(self, *args, **kwargs):
        return self._data_loader(self.dataset["test"], shuffle=False)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.bsz,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
