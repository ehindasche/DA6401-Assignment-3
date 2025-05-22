from libraries import *
from data_processing import *
from attention_RNN import *


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
import csv
from torch.utils.data import Subset
import torch.nn.functional as F

def train_with_sweep():
    wandb.init()
    config = wandb.config

    # Load data
    train_data, dev_data, test_data = load_data('hi')
    latin_vocab, devanagari_vocab = create_vocab(pd.concat([train_data, dev_data]))
    train_dataset = TransliterationDataset(train_data, latin_vocab, devanagari_vocab)
    dev_dataset = TransliterationDataset(dev_data, latin_vocab, devanagari_vocab)
    test_dataset = TransliterationDataset(test_data, latin_vocab, devanagari_vocab)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(len(latin_vocab), config.enc_emb_dim, config.hid_dim,
                      dropout=config.enc_dropout, cell_type=config.cell_type, num_layers=config.num_layers).to(device)

    decoder = AttnDecoder(len(devanagari_vocab), config.dec_emb_dim, config.hid_dim,
                          dropout=config.dec_dropout, cell_type=config.cell_type, num_layers=config.num_layers).to(device)

    model = AttnSeq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_val_word_accuracy = 0
    for epoch in range(config.epochs):
        train_metrics = train(model, train_loader, optimizer, criterion, config.clip, latin_vocab, devanagari_vocab, tf_ratio=config.tf_ratio)
        val_metrics = evaluate(model, dev_loader, criterion, latin_vocab, devanagari_vocab)

        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_char_acc': train_metrics['char_accuracy'],
            'train_word_acc': train_metrics['word_accuracy'],
            'val_loss': val_metrics['loss'],
            'val_char_acc': val_metrics['char_accuracy'],
            'val_word_acc': val_metrics['word_accuracy']
        })
        # print(val_metrics['attentions'])
        # if val_metrics['word_accuracy'] > best_val_word_accuracy:
        #     best_val_word_accuracy = val_metrics['word_accuracy']
        #     torch.save(model.state_dict(), 'best_model.pt')


# === Sweep Config ===
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_word_acc',
        'goal': 'maximize'
    },
    'parameters': {
        'batch_size': {'values': [64, 128]},
        'enc_emb_dim': {'values': [256, 512]},
        'dec_emb_dim': {'values': [256, 512]},
        'hid_dim': {'values': [512, 1024]},
        'enc_dropout': {'values':[0.3, 0.2]},
        'dec_dropout': {'values': [0.1, 0.2]},
        'epochs': {'value': 4},
        'clip': {'value': 1},
        'lr': {'values':[0.00005,0.0005, 0.001]},
        'cell_type': {'values': ['LSTM']},
        'num_layers':{'values': [2,3]},
        'tf_ratio':{'values':[0.2, 0.5, 0.8]}
    }
}


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='DA6401-Assignment3-Attention')
    wandb.agent(sweep_id, function=train_with_sweep, count = 10)