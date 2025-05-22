from libraries import *
from data_processing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout=0.1, cell_type='GRU'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.upper()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

        rnn_class = getattr(nn, self.cell_type)
        self.rnn = rnn_class(
            embedding_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

# Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        if hidden.dim() == 3:
            hidden = hidden[-1]  # Use top layer hidden state

        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

# Decoder with Attention
class AttnDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, dropout=0.1, cell_type='GRU'):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)

        rnn_class = getattr(nn, self.cell_type)
        self.rnn = rnn_class(
            embedding_size + hidden_size, hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True
        )

        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x, hidden, encoder_outputs):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))

        if self.cell_type == 'LSTM':
            context_hidden = hidden[0]
        else:
            context_hidden = hidden

        attn_weights = self.attention(context_hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        output = output.squeeze(1)
        context = context.squeeze(1)
        output = self.out(torch.cat((output, context), dim=1))

        return output, hidden, attn_weights.squeeze(1)

# Seq2Seq with Attention
class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, cell_type='GRU'):
        super(AttnSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.cell_type = cell_type.upper()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.size(1)).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        if self.cell_type == 'LSTM':
            input_hidden = (hidden[0], hidden[1])
        else:
            input_hidden = hidden

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, input_hidden, attention = self.decoder(input, input_hidden, encoder_outputs)
            outputs[:, t] = output
            attentions[:, t] = attention

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs, attentions

def train(model, iterator, optimizer, criterion, clip, latin_vocab, devanagari_vocab, tf_ratio=0.5):
    model.train()
    epoch_loss = 0
    char_correct = 0
    char_total = 0
    
    devanagari_reverse_vocab = {v: k for k, v in devanagari_vocab.items()}
    sample_indices = random.sample(range(len(iterator.dataset)), min(100, len(iterator.dataset)))
    sample_dataset = Subset(iterator.dataset, sample_indices)
    sample_loader = DataLoader(sample_dataset, batch_size=iterator.batch_size, collate_fn=collate_fn, shuffle=False)
    
    for src, trg in tqdm(iterator, desc="Training"):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output, _ = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Training character accuracy
        preds = output.argmax(1)
        mask = (trg != 0) & (trg != 1)
        char_correct += ((preds == trg) & mask).sum().item()
        char_total += mask.sum().item()
    
    # Calculate word accuracy on sample
    word_correct = 0
    word_total = 0
    
    for src, trg in sample_loader:
        src, trg = src.to(device), trg.to(device)
        output, _ = model(src, trg, teacher_forcing_ratio=tf_ratio)
        preds = output.argmax(2)
        
        for i in range(src.size(0)):
            trg_word = ''.join([devanagari_reverse_vocab.get(idx.item(), '') 
                             for idx in trg[i] 
                             if idx.item() not in {0, 1, 2, 3}])
            pred_word = ''.join([devanagari_reverse_vocab.get(idx.item(), '') 
                               for j, idx in enumerate(preds[i]) 
                               if j > 0 and idx.item() not in {0, 1, 2, 3}])
            if trg_word == pred_word:
                word_correct += 1
            word_total += 1
    
    return {
        'loss': epoch_loss / len(iterator),
        'char_accuracy': char_correct / char_total,
        'word_accuracy': word_correct / max(1, word_total)
    }

def decode_tensor_to_str(tensor, reverse_vocab, skip_tokens={0,1,2,3}):
    chars = []
    for idx in tensor:
        idx_val = idx.item()
        if idx_val not in skip_tokens:
            chars.append(reverse_vocab.get(idx_val, '?'))  # '?' for unknowns
    return ''.join(chars)
    
def evaluate(model, dataloader, criterion, latin_vocab, devanagari_vocab):
    model.eval()
    total_loss = 0
    total_chars = 0
    correct_chars = 0
    correct_words = 0
    total_words = 0

    all_predictions = []
    devanagari_reverse_vocab = {v: k for k, v in devanagari_vocab.items()}
    latin_reverse_vocab = {v: k for k, v in latin_vocab.items()}

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            output, attentions = model(src, trg, 0)  # no teacher forcing
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            total_loss += loss.item()

            preds = output.argmax(-1)  # [batch, trg_len]

            for i in range(trg.shape[0]):
                src_word_tensor = src[i]
                trg_word_tensor = trg[i]
                pred_tensor = preds[i][1:]  # exclude <sos>
                attn_tensor = attentions[i][1:len(pred_tensor)+1]  # shape: [trg_len-1, src_len]

                src_word = decode_tensor_to_str(src_word_tensor, latin_reverse_vocab)
                trg_word = decode_tensor_to_str(trg_word_tensor, devanagari_reverse_vocab)
                pred_word = decode_tensor_to_str(pred_tensor, devanagari_reverse_vocab)

                attention_matrix = attn_tensor[:len(pred_word), :len(src_word)].cpu().tolist()

                all_predictions.append({
                    'source': src_word,
                    'target': trg_word,
                    'prediction': pred_word,
                    'correct': trg_word == pred_word,
                    'attention': attention_matrix  # list of [trg_char_i -> attention over src_char_j]
                })

                total_words += 1
                if trg_word == pred_word:
                    correct_words += 1

                for j in range(1, trg.shape[1]):
                    if trg[i][j] != 0:
                        total_chars += 1
                        if trg[i][j] == preds[i][j]:
                            correct_chars += 1

    return {
        'loss': total_loss / len(dataloader),
        'char_accuracy': correct_chars / total_chars,
        'word_accuracy': correct_words / total_words,
        'predictions': all_predictions  # renamed from 'attentions'
    }