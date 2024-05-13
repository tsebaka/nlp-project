import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class EssayDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        dataframe,
        train=True
    ):
        self.config = config
        self.dataframe = dataframe
        self.texts = dataframe["full_text"].values
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, use_fast=False)
        
        self.labels = None
        if config.main.target_col in dataframe.columns and train:
            self.labels = dataframe[config.main.target_col].values - 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            **(self.config.tokenizer)
        )
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return inputs['input_ids'], inputs['attention_mask'], label
        return inputs['input_ids'], inputs['attention_mask']


def get_train_dataloader(config, dataframe):
    dataset = EssayDataset(config, dataframe)
    dataloader = DataLoader(
        dataset,
        batch_size=config.main.train_batch_size,
        num_workers=config.main.num_workers,
        shuffle=True,
        drop_last=True
    )
    return dataloader


def get_valid_dataloader(config, dataframe):
    dataset = EssayDataset(config, dataframe)
    dataloader = DataLoader(
        dataset,
        batch_size=config.main.valid_batch_size,
        num_workers=config.main.num_workers,
        shuffle=True,
        drop_last=False,
    )
    return dataloader


def get_test_dataloader(config, dataframe):
    dataset = EssayDataset(config, dataframe)
    dataloader = DataLoader(
        dataset,
        batch_size=config.main.test_batch_size,
        num_workers=config.main.num_workers,
        shuffle=False,
        drop_last=False
    )
    return dataloader