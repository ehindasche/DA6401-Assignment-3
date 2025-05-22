# DL Assignment 3 - Attention-based Sequence-to-Sequence Transliteration Model

This repository contains the implementation of a neural sequence-to-sequence transliteration model using **Bahdanau Attention**. The model supports multiple RNN cell types (LSTM, GRU, or vanilla RNN), and is designed for transliterating words from a source language to a target language.

---

## 📌 Features

- Encoder-decoder architecture using Bahdanau attention
- Configurable RNN cell type (LSTM/GRU/RNN)
- Custom training loop using teacher forcing
- Loss masking for variable-length sequences
- Configurable hyperparameters via argparse or direct modification

---

## 🧠 Model Architecture

- **Encoder**: Single-layer RNN (LSTM/GRU/RNN), returns hidden states
- **Attention**: Additive Bahdanau-style attention
- **Decoder**: RNN with context vector + embedding + previous output
- **Output**: Softmax over target vocabulary

---
## 📁 Project Structure

```
dl-assignment-3/
├── transliteration_model.py     # Main training and evaluation script
├── visualizations/              # contains animation images
├── src/                         # py files
├── heatmaps/                    # contains saved heatmaps
└── dl-assignment-3.ipynb        # Jupyter Notebook version
```
