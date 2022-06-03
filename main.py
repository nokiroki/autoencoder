import numpy as np
import pandas as pd
import os
from pytorch_lightning import Trainer
from torchsummary import summary

from src import Conv1dAutoEncoder, TransactionDataModule, TransactionAnomalyDataModule, LSTMAutoEncoder
from src.datamodules.cae_datamodule import TransactionsDataset


def create_info_file():
    df = pd.read_csv('data/transactions_train.csv')
    df_names = pd.DataFrame(np.unique(df.client_id))
    # df_names.to_csv('data/data/info.csv')

    # print(df_names.head())
    print(df_names.shape)


def test_lstm_network():
    model = LSTMAutoEncoder(40, 3)
    trainer = Trainer(gpus=1, max_epochs=20)
    dm = TransactionDataModule('data\\data\\transactions_40_train.csv',
                               'data\\data\\transactions_40_test.csv',
                               drop_time=True)
    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_cae_network():
    model = Conv1dAutoEncoder(3, 8)
    trainer = Trainer(gpus=1, max_epochs=20)
    dm = TransactionDataModule('data\\data\\transactions_40_train.csv',
                               'data\\data\\transactions_40_test.csv',
                               drop_time=True)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def get_summary(model, device):
    model = model.to(device)
    return summary(model, (3, 40), batch_size=1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test_lstm_network()
    # test_cae_network()
    get_summary(LSTMAutoEncoder(40, 3), 'cuda')
