import sys
import os
sys.path.insert(0, "/home/jovyan/.local/share/virtualenvs/ptls-experiments-w-dEu3oS/lib/python3.8/site-packages")
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import hydra
import torch
import logging

from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from src.data.preprocessing import make_folds
from src.dataset.dataset import get_train_dataloader, get_valid_dataloader 
from src.models.utils import get_model
from src.criterion.criterion import get_criterion
from src.optimizer.optimizer import get_optimizer
from src.metric.metric import quadratic_weighted_kappa
from src.validation.validation import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def five_folds_training(config, dataset, logger):
    for fold in range(config.main.n_folds):
        train_folds = dataset[dataset["fold"] != fold]
        valid_folds = dataset[dataset["fold"] == fold]
        
        train_loop(config, train_folds, valid_folds, fold, logger)

def train_loop(
    config,
    train_folds,
    valid_folds,
    fold,
    logger,
):
    train_dataloader = get_train_dataloader(config, train_folds)
    valid_dataloader = get_valid_dataloader(config, valid_folds)
    
    model = get_model(config)
    model.to(device)
    
    criterion = get_criterion(config)
    optimizer = get_optimizer(model)
    
    logging.info("Start training...")
    for epoch in range(config.train_params.num_train_epochs):
        model.train()
        train_loss = []
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids, attention_mask, labels = batch
            output = model.forward(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            
            loss = torch.sqrt(criterion(output["logits"].view(-1).to(device), labels.to(device).float()))
            train_loss.append(loss.item())
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step + 1) % config.train_params.check_val_every == 0:
                evaluate(config, model, valid_dataloader, criterion, step + len(train_dataloader) * epoch, logger)
              
        logging.info("epoch:", epoch)
        logging.info("train loss:", np.sum(train_loss) / len(train_dataloader))
        # logging.info("Saving model...")
        # model.save_pretrained(f"models/roberta/roberta-large-fold-{fold}")


@hydra.main(version_base=None, config_path="config", config_name="deberta-v3-large")
def main(config):
    if config.main.extended_data:
        dataset = make_folds(config, pd.read_csv(config.main.extended_path))
    else:
        dataset = make_folds(config, pd.read_csv(config.main.main_train_path))
    
    logger = SummaryWriter()
    five_folds_training(config, dataset, logger)

if __name__ == '__main__':
    main()