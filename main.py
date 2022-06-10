from pytorch_lightning import Trainer, loggers
from torchsummary import summary

from src import Conv1dAutoEncoder, TransactionDataModule, LSTMAutoEncoder, Conv1dEmbedAutoEncoder


def test_lstm_network(train_dataset, test_dataset):
    model = LSTMAutoEncoder(40, 3)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'lstm')
    trainer = Trainer(gpus=1, max_epochs=20, logger=logger)
    dm = TransactionDataModule(train_dataset, test_dataset, drop_time=False)
    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_cae_network(train_dataset, test_dataset):
    model = Conv1dAutoEncoder(4, 8)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'cae')
    trainer = Trainer(gpus=1, max_epochs=50, logger=logger)
    dm = TransactionDataModule(train_dataset, test_dataset, drop_time=False, with_anomalies=True)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def test_cae_with_embed_network(train_dataset, test_dataset):
    model = Conv1dEmbedAutoEncoder(4, 8)
    logger = loggers.TensorBoardLogger('lightning_logs_new', 'cae_with_embed')
    trainer = Trainer(gpus=1, max_epochs=20, logger=logger)
    dm = TransactionDataModule(train_dataset, test_dataset)

    trainer.fit(model, dm)
    trainer.test(model, dm)


def get_summary(model, device):
    model = model.to(device)
    return summary(model, (3, 40), batch_size=1)


if __name__ == '__main__':
    # test_lstm_network()
    test_cae_with_embed_network('data\\data\\exp3_train_small.csv', 'data\\data\\exp3_test.csv')
