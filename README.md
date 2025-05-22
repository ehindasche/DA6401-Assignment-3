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
├── visualizations/              # contains animation images
├── src/                         # py files
├── heatmaps/                    # contains saved heatmaps
├── predictions_vanilla.csv      # vanilla rnn predictions on test set
├── predictions_attention.csv    # attention based rnn predictions on test set                
└── dl-assignment-3.ipynb        # Jupyter Notebook version
```
---
## How to run?

- First run the vanilla_RNN.py through bash.
```bash
python "src/vanilla_RNN.py"
```
This will give us the sweeep results over the Vanilla RNN model.

- Then run the vanilla_evaluation_test_set.py.
```bash
python "src/vanilla_evaluation_test_set.py"
```

- Now run the attention_RNN.py

```bash
python "src/attention_RNN.py"
```

- Now run attention_RNN_sweep.py
```bash
python "src/attention_RNN_sweep.py"
```
- Now run heatmaps.py
```bash
python "src/heatmaps.py"
```
- Now run visualizations.py
```bash
python "src/visualizations.py"
```
drive link for sliding visualizations: https://drive.google.com/file/d/1gHWrV1Jd-N7d8MG4QIZWPnx_EV-apg33/view?usp=sharing
---
- Heatmaps folder contains the heatmap images for 9 words (showing how attention is spread across different characters)
- Visualizations folder contains the images of the 4 words for which we demonstrated the sliding animation of attention weights over different characters
- predictions_vanilla.csv contains test set predictions for vanilla model
- predictions_attention.csv containts test set predictions for attention based model
