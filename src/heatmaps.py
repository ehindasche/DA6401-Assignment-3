from libraries import *

# --- Define devanagari_font_prop *before* this function ---
# This part goes *before* any functions that use it, typically near your other imports.
# It should be based on the successful font loading from your previous steps.

font_path = '/kaggle/input/notosans/NotoSansDevanagari-VariableFont_wdthwght.ttf' # **MAKE SURE THIS IS YOUR CORRECT PATH**

devanagari_font_prop = None # Initialize to None
try:
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path) # Add it to Matplotlib's font manager
        devanagari_font_prop = fm.FontProperties(fname=font_path)
        # Set global rcParams too, as it helps for other elements
        plt.rcParams['font.sans-serif'] = [devanagari_font_prop.get_name(), 'FreeSans', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        print(f"Matplotlib configured using specific font path: {font_path}")
    else:
        print(f"Error: Font file not found at {font_path}. Devanagari text may not render correctly.")
except Exception as e:
    print(f"Error loading or configuring Devanagari font: {e}")
    devanagari_font_prop = None # Ensure it's None if there was an error
# -----------------------------------------------------------


def plot_attention_heatmap(sample):
    source = list(sample['source'])
    target = list(sample['prediction']) # 'target' is the predicted Devanagari word
    attention = sample['attention']  # Shape: [target_len, source_len]

    # Dynamically adjust figure size for better readability
    fig_width = max(8, len(source) * 0.5)
    fig_height = max(6, len(target) * 0.5)

    plt.figure(figsize=(fig_width, fig_height))

    # Create the heatmap
    sns.heatmap(attention, xticklabels=source, yticklabels=target, cmap='viridis')

    # Apply the Devanagari font to elements that should display Devanagari
    if devanagari_font_prop:
        # X-axis label (Source - Latin)
        plt.xlabel('Source (Latin)', fontsize=12) # No font_properties here if Latin

        # Y-axis label (Predicted - Devanagari)
        plt.ylabel('Predicted (Devanagari)', fontsize=12, **{'fontproperties': devanagari_font_prop})

        # Title (contains both Latin and Devanagari)
        plt.title(f"Attention Heatmap\nSource: {sample['source']} | Prediction: {sample['prediction']}", fontsize=14, **{'fontproperties': devanagari_font_prop})

        # Apply font to Y-axis tick labels (the predicted Devanagari characters)
        for tick_label in plt.gca().get_yticklabels():
            tick_label.set_fontproperties(devanagari_font_prop)

        # Apply font to X-axis tick labels (source characters - generally Latin, but good to ensure if needed)
        # You might not need this if source characters are always Latin and your default font handles them.
        # However, if your source characters could ever be complex or have mixed scripts, it's safer.
        for tick_label in plt.gca().get_xticklabels():
            # You could add a check here if characters are Devanagari:
            # if any('\u0900' <= char <= '\u097F' for char in tick_label.get_text()):
            tick_label.set_fontproperties(devanagari_font_prop)

    else:
        # Fallback if font could not be loaded or prop is None
        plt.xlabel('Source (Latin)', fontsize=12)
        plt.ylabel('Predicted (Devanagari)', fontsize=12)
        plt.title(f"Attention Heatmap\nSource: {sample['source']} | Prediction: {sample['prediction']}", fontsize=14)

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() # Or wandb.Image(fig) if logging to wandb

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import random
import csv
from torch.utils.data import Subset
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # Make sure this is imported at the top of your script
import seaborn as sns
import os

# ---- Set Seeds for Reproducibility ----
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---- Best Hyperparameters ----
best_hparams = {
    'batch_size': 128,
    'hidden_size': 1024,
    'embedding_dim': 512,
    'dropout': 0.2,
    'cell_type': 'LSTM',
    'num_layers': 2,
    'lr': 0.001,
    'epochs': 4,
    'tf_ratio': 0.8
}

# ---- Data Loading & Vocabulary ----
train_data, dev_data, test_data = load_data('hi')
latin_vocab, devanagari_vocab = create_vocab(pd.concat([train_data, dev_data]))

train_dataset = TransliterationDataset(train_data, latin_vocab, devanagari_vocab)
dev_dataset = TransliterationDataset(dev_data, latin_vocab, devanagari_vocab)
test_dataset = TransliterationDataset(test_data, latin_vocab, devanagari_vocab)

train_loader = DataLoader(train_dataset, batch_size=best_hparams['batch_size'], shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=best_hparams['batch_size'], shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=best_hparams['batch_size'], shuffle=False, collate_fn=collate_fn)

# ---- Model Definition ----
encoder = Encoder(len(latin_vocab), best_hparams['embedding_dim'], best_hparams['hidden_size'],
                  dropout=best_hparams['dropout'], cell_type=best_hparams['cell_type'],
                  num_layers=best_hparams['num_layers']).to(device)

decoder = AttnDecoder(len(devanagari_vocab), best_hparams['embedding_dim'], best_hparams['hidden_size'],
                      dropout=best_hparams['dropout'], cell_type=best_hparams['cell_type'],
                      num_layers=best_hparams['num_layers']).to(device)

model = AttnSeq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters(), lr=best_hparams['lr'])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ---- Training Loop ----
for epoch in range(best_hparams['epochs']):
    train_metrics = train(model, train_loader, optimizer, criterion, epoch + 1, latin_vocab, devanagari_vocab, tf_ratio = best_hparams['tf_ratio'])
    val_metrics = evaluate(model, dev_loader, criterion, latin_vocab, devanagari_vocab)
    print(f"Epoch {epoch+1} Validation - Loss: {val_metrics['loss']:.4f} | Word Acc: {val_metrics['word_accuracy']:.4f}")

# ---- Final Evaluation on Test Set ----
test_metrics = evaluate(model, test_loader, criterion, latin_vocab, devanagari_vocab)
print(f"\nTest Set - Loss: {test_metrics['loss']:.4f} | Word Accuracy: {test_metrics['word_accuracy']:.4f} | Char accuracy: {test_metrics['char_accuracy']:.4f}")

# ---- Plot First 9 Test Predictions ----
if len(test_metrics['predictions']) < 9:
    print("Not enough test predictions to plot 9 heatmaps.")
else:
    for i in range(9):
        sample = test_metrics['predictions'][i]
        plot_attention_heatmap(sample)

prediction_data = [{
    'source': sample['source'],
    'target': sample['target'],
    'prediction': sample['prediction'],
    'correct': sample['correct']
} for sample in test_metrics['predictions']]

df = pd.DataFrame(prediction_data)
df.to_csv('test_predictions.csv', index=False)
print("Saved test predictions to 'test_predictions.csv'")