from libraries import *
from data_processing import *

class Encoder(nn.Module):
    # takes input as input_size, embedding size (dim of xt), hidden_size (dim of st),
    # num_layers = output of the RNN is feeded as input to another RNN (multi-layer)
    # cell_type = LSTM, Vanilla RNN or GRU
    # dropout for weights of the RNN (U, W)
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, cell_type='LSTM', dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        # embedding is initialized and is trained via backprop
        # converts token IDs from vocab into dense vectors
        # self.embedding is the initialization of the weight matrix for it
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # Vanilla RNN
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # e.g - x can be ['angakor', 'pair', 'haath']. 
        # Here seq_len is the length of the strings 'angakor', 'haath', etc. (i.e 7, 5, etc)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embedding_size)
        
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded) # input passed
            return outputs, hidden, cell
            # Passes the embedded input through the LSTM.
            # 'outputs': Contains the hidden state output for each time step of the *last* layer.
            #            Shape: (batch_size, seq_len, hidden_size).
            # 'hidden': The final hidden state for each layer.
            #           Shape: (num_layers * num_directions, batch_size, hidden_size).
            # 'cell': The final cell state for each layer (specific to LSTM).
            #         Shape: (num_layers * num_directions, batch_size, hidden_size).
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden, None

        # Returns all three. For Seq2Seq, 'hidden' and 'cell' are typically passed to the decoder.

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers, cell_type='LSTM', dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        else:  # Vanilla RNN
            self.rnn = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden, cell=None):
        # x shape: (batch_size, 1)
        x = x.unsqueeze(1)  # (batch_size, 1)
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embedding_size)
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, hidden = self.rnn(embedded, hidden)
        
        prediction = self.fc(output.squeeze(1))  # (batch_size, output_size)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        # source: Input batch of source sequences (e.g., Latin words). Shape: (batch_size, source_seq_len).
        # target: Input batch of target sequences (e.g., Devanagari words). Shape: (batch_size, target_seq_len).
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.fc.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # Encoder forward pass
        encoder_outputs, hidden, cell = self.encoder(source)
        
        # First input to decoder is <SOS> token
        input = target[:, 0]
        
        for t in range(1, target_len):
            # Decoder forward pass
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Store predictions
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            
            # Get the next input
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1
        
        return outputs

# Define the training function with all required imports
def train(model, iterator, optimizer, criterion, clip, latin_vocab, devanagari_vocab):
    model.train()
    epoch_loss = 0
    char_correct = 0
    char_total = 0
    
    # Create reverse vocabulary
    devanagari_reverse_vocab = {v: k for k, v in devanagari_vocab.items()}
    
    # Get random sample indices for word accuracy calculation
    all_indices = list(range(len(iterator.dataset)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:min(100, len(all_indices))]
    
    for src, trg in tqdm(iterator, desc="Training"):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        
        # Calculate loss
        output_dim = output.shape[-1]
        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)
        loss = criterion(output_flat, trg_flat)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Training character accuracy
        preds = output.argmax(2)
        mask = (trg != 0) & (trg != 1)
        char_correct += ((preds == trg) & mask).sum().item()
        char_total += mask.sum().item()
    
    # Calculate word accuracy on sample
    word_correct = 0
    word_total = 0
    sample_dataset = Subset(iterator.dataset, sample_indices)
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=iterator.batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )
    
    for src, trg in sample_loader:
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=0)
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

def evaluate(model, iterator, criterion, latin_vocab, devanagari_vocab):
    model.eval()
    epoch_loss = 0
    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0
    
    # Create reverse vocabulary inside the function
    devanagari_reverse_vocab = {v: k for k, v in devanagari_vocab.items()}
    
    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating"):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
            
            # Character accuracy
            preds = output.argmax(2)
            mask = (trg != 0) & (trg != 1)
            char_correct += ((preds == trg) & mask).sum().item()
            char_total += mask.sum().item()
            
            # Word accuracy
            for i in range(src.size(0)):
                # Get target word
                trg_word = ''.join([devanagari_reverse_vocab.get(idx.item(), '') 
                                  for idx in trg[i] 
                                  if idx.item() not in {0, 1, 2, 3}])
                
                # Get predicted word
                pred_word = ''.join([devanagari_reverse_vocab.get(idx.item(), '') 
                                   for j, idx in enumerate(preds[i]) 
                                   if j > 0 and idx.item() not in {0, 1, 2, 3}])
                
                if trg_word == pred_word:
                    word_correct += 1
                word_total += 1
    
    return {
        'loss': epoch_loss / len(iterator),
        'char_accuracy': char_correct / char_total if char_total > 0 else 0,
        'word_accuracy': word_correct / word_total if word_total > 0 else 0
    }

sweep_config = {
    'method': 'bayes',  # Bayesian optimization is good for this search space
    'metric': {
        'name': 'val_word_accuracy',  # Now optimizing for word-level accuracy
        'goal': 'maximize'
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,
        'eta': 2
    },
    'parameters': {
        'embedding_size': {
            'values': [32, 64, 128, 256],  # Added 256, removed 16 (too small)
            'distribution': 'categorical'
        },
        'hidden_size': {
            'values': [128, 256, 512],  # Increased ranges for better representation
            'distribution': 'categorical'
        },
        'num_layers': {
            'values': [1, 2],  # Removed 3 (rarely helps for this task)
            'distribution': 'categorical'
        },
        'cell_type': {
            'values': ['GRU', 'LSTM', 'RNN'],  # Removed vanilla RNN (known to be worse)
            'distribution': 'categorical'
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3, 0.4],  # Added more options
            'distribution': 'categorical'
        },
        'learning_rate': {
            'min': 0.0005,  # Narrowed range based on typical good values
            'max': 0.005,
            'distribution': 'log_uniform_values'
        },
        'batch_size': {
            'values': [64, 128, 256],  # Increased upper range
            'distribution': 'categorical'
        },
        'teacher_forcing_ratio': {  # Added new important parameter
            'min': 0.5,
            'max': 0.9,
            'distribution': 'uniform'
        }
    }
}

def sweep_train():
    wandb.init()
    
    try:
        config = wandb.config
        
        # Load data
        train_data, dev_data, test_data = load_data('hi')
        latin_vocab, devanagari_vocab = create_vocab(pd.concat([train_data, dev_data]))
        
        # Create datasets
        train_dataset = TransliterationDataset(train_data, latin_vocab, devanagari_vocab)
        val_dataset = TransliterationDataset(dev_data, latin_vocab, devanagari_vocab)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize model
        encoder = Encoder(
            input_size=len(latin_vocab),
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        ).to(device)
        
        decoder = Decoder(
            output_size=len(devanagari_vocab),
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            cell_type=config.cell_type,
            dropout=config.dropout
        ).to(device)
        
        model = Seq2Seq(encoder, decoder, device).to(device)
        
        # Initialize optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        best_val_word_accuracy = 0
        for epoch in range(10):
            train_metrics = train(
                model, train_loader, optimizer, criterion,
                clip=1, latin_vocab=latin_vocab,
                devanagari_vocab=devanagari_vocab
            )
            
            val_metrics = evaluate(
                model, val_loader, criterion,
                latin_vocab=latin_vocab,
                devanagari_vocab=devanagari_vocab
            )
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_char_accuracy': train_metrics['char_accuracy'],
                'train_word_accuracy': train_metrics['word_accuracy'],
                'val_loss': val_metrics['loss'],
                'val_char_accuracy': val_metrics['char_accuracy'],
                'val_word_accuracy': val_metrics['word_accuracy']
            })
            
            # Save best model
            if val_metrics['word_accuracy'] > best_val_word_accuracy:
                best_val_word_accuracy = val_metrics['word_accuracy']
                torch.save(model.state_dict(), 'best_model.pt')
                wandb.save('best_model.pt')
        
        wandb.log({'best_val_word_accuracy': best_val_word_accuracy})
        
    except Exception as e:
        wandb.log({"error": str(e)})
        raise e
        
# Run the sweep
sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment3")
wandb.agent(sweep_id, function=sweep_train)