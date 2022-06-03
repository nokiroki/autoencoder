import os
from abc import ABC
from typing import Optional
import math

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import T_co
from sklearn.preprocessing import StandardScaler


class TransactionsDatasetIndex(Dataset):

    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.info = pd.read_csv(os.path.join(data_folder, 'info.csv'), index_col=[0])

    def __getitem__(self, item):
        return torch.from_numpy(
            pd.read_csv(
                os.path.join(self.data_folder, f'{self.info.iloc[item]["idx"]}.csv'),
                index_col=[0]).values)

    def __len__(self):
        return self.info.shape[0]


class TransactionsDataset(Dataset):

    def __init__(self, data_file, max_length: int = 40, drop_time: bool = False):
        super(TransactionsDataset, self).__init__()
        self.max_length = max_length
        self.data_file = data_file
        self.drop_time = drop_time
        self.all_data = pd.read_csv(data_file, index_col=[0])
        if drop_time:
            self.all_data.drop(columns=['trans_date'], axis=1, inplace=True)
        self.sc = StandardScaler().fit(self.all_data)

    def __getitem__(self, index) -> T_co:
        sample = self.sc.transform(self.all_data.iloc[index * self.max_length:(index + 1) * self.max_length]).T
        # Scaling time column
        if not self.drop_time:
            sample[1, :] -= sample[1, 0]
        return torch.tensor(sample).float()

    def __len__(self):
        return math.floor(self.all_data.shape[0] / self.max_length)


class TransactionsDatasetWithAnomaly(Dataset):

    def __init__(self, data_file, max_length: int = 20):
        super(TransactionsDatasetWithAnomaly, self).__init__()
        self.max_length = max_length
        self.data_file = data_file
        self.all_data = pd.read_csv(data_file, index_col=[0]).drop(labels=['Class'], axis=1)

    def __getitem__(self, index) -> T_co:
        tensor = torch.tensor(
            self.all_data.iloc[index].values
        ).float()
        tensor.unsqueeze_(0)
        return tensor

    def __len__(self):
        return self.all_data.shape[0]


class TransactionAnomalyDataModule(LightningDataModule):

    def __init__(self, data_folder='.\\', batch_size=64):
        super(TransactionAnomalyDataModule, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.test = None
        self.train = None
        self.val = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in [None, 'fit']:
            train_val = TransactionsDatasetWithAnomaly(os.path.join(self.data_folder, 'creditcard_norm_train.csv'))
            l = len(train_val)
            self.train, self.val = random_split(train_val, (int(l * 0.8), l - int(l * 0.8)))

        if stage in [None, 'test']:
            self.test = TransactionsDataset(os.path.join(self.data_folder, 'creditcard_norm_test.csv'))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass


class TransactionDataModule(LightningDataModule):

    def __init__(self, data_train_file='.\\', data_test_file='.\\', batch_size=64, max_length=40, drop_time=False):
        super(TransactionDataModule, self).__init__()
        self.data_train_file = data_train_file
        self.data_test_file = data_test_file
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_time = drop_time
        self.test = None
        self.train = None
        self.val = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in [None, 'fit']:
            train_val = TransactionsDataset(self.data_train_file, self.max_length, self.drop_time)
            l = len(train_val)
            self.train, self.val = random_split(train_val, (int(l * 0.8), l - int(l * 0.8)))

        if stage in [None, 'test']:
            self.test = TransactionsDataset(self.data_test_file, self.max_length, self.drop_time)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test,
                          batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val,
                          batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def collate_fn(self, data):
        data = pad_sequence(data, batch_first=True).reshape(self.batch_size, 4, -1).float()

        # print(type(data))
        # print(len(data))
        # print(data[0].shape)
        # print(data[1].shape)

        return data
