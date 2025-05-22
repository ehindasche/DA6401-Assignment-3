from vanilla_RNN import *
from libraries import *
from data_processing import *

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def decode_tensor_to_str(tensor, reverse_vocab, skip_tokens={0,1,2,3}):
    chars = []
    for idx in tensor:
        idx_val = idx.item()
        if idx_val not in skip_tokens:
            chars.append(reverse_vocab.get(idx_val, '?'))  # '?' for unknowns
    return ''.join(chars)

def evaluate_test_set(model, test_data, latin_vocab, devanagari_vocab, save_dir='predictions_vanilla'):
    # Create test dataset and loader
    test_dataset = TransliterationDataset(test_data, latin_vocab, devanagari_vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Create reverse vocabularies
    devanagari_reverse_vocab = {v: k for k, v in devanagari_vocab.items()}
    # print("Sample Devanagari Vocab:", list(devanagari_vocab.items())[:10])
    # print("Sample Reverse Vocab:", list(devanagari_reverse_vocab.items())[:10])
    # Evaluate
    model.eval()
    all_predictions = []
    char_predictions = []
    char_targets = []
    
    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0)
            preds = output.argmax(2)
            
            for i in range(src.size(0)):
                src_word = decode_tensor_to_str(src[i], {v: k for k, v in latin_vocab.items()})
                trg_word = decode_tensor_to_str(trg[i], devanagari_reverse_vocab)
                pred_word = decode_tensor_to_str(preds[i][1:], devanagari_reverse_vocab)
            
                all_predictions.append({
                    'source': src_word,
                    'target': trg_word,
                    'prediction': pred_word,
                    'correct': trg_word == pred_word
                })
                
                # For confusion matrix
                for j, idx in enumerate(trg[i]):
                    if idx.item() not in {0, 1, 2, 3}:  # Skip special tokens
                        char_targets.append(devanagari_reverse_vocab.get(idx.item(), ''))
                        if j < preds[i].shape[0]:
                            char_predictions.append(devanagari_reverse_vocab.get(preds[i][j].item(), ''))
    
    # Calculate accuracy
    correct = sum(p['correct'] for p in all_predictions)
    total = len(all_predictions)
    accuracy = correct / total
    
    # Save predictions
    os.makedirs(save_dir, exist_ok=True)
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)
    
    # Generate confusion matrix
    chars = sorted(set(char_targets + char_predictions))
    cm = confusion_matrix(char_targets, char_predictions, labels=chars)
    
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=chars, yticklabels=chars)
    plt.title('Character-level Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return accuracy, predictions_df, cm

# Manually configure the best model
best_config = {
    'batch_size': 64,
    'cell_type': 'LSTM',
    'dropout': 0.4,
    'embedding_size': 128,
    'hidden_size': 256,
    'learning_rate': 0.001326,
    'num_layers': 2,
    'teacher_forcing_ratio': 0.58607
}

train_data, dev_data, test_data = load_data('hi')
latin_vocab, devanagari_vocab = create_vocab(pd.concat([train_data, dev_data]))


# Create datasets
train_dataset = TransliterationDataset(train_data, latin_vocab, devanagari_vocab)
val_dataset = TransliterationDataset(dev_data, latin_vocab, devanagari_vocab)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=best_config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=best_config['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)

# Initialize model with manual configuration
encoder = Encoder(
    input_size=len(latin_vocab),
    embedding_size=best_config['embedding_size'],
    hidden_size=best_config['hidden_size'],
    num_layers=best_config['num_layers'],
    cell_type=best_config['cell_type'],
    dropout=best_config['dropout']
).to(device)

decoder = Decoder(
    output_size=len(devanagari_vocab),
    embedding_size=best_config['embedding_size'],
    hidden_size=best_config['hidden_size'],
    num_layers=best_config['num_layers'],
    cell_type=best_config['cell_type'],
    dropout=best_config['dropout']
).to(device)

manual_model = Seq2Seq(encoder, decoder, device).to(device)

# Train the model (assuming you have train_loader and val_loader)
optimizer = optim.Adam(manual_model.parameters(), lr=best_config['learning_rate'])
criterion = nn.CrossEntropyLoss(ignore_index=0)

best_val_accuracy = 0
for epoch in range(10):  # Train for 10 epochs
    train_metrics = train(
        manual_model, train_loader, optimizer, criterion,
        clip=1, latin_vocab=latin_vocab,
        devanagari_vocab=devanagari_vocab
    )
    
    val_metrics = evaluate(
        manual_model, val_loader, criterion,
        latin_vocab=latin_vocab,
        devanagari_vocab=devanagari_vocab
    )
    
    print(f"Epoch {epoch+1}:")
    print(f"Train Loss: {train_metrics['loss']:.4f} | Word Acc: {train_metrics['word_accuracy']:.2%}")
    print(f"Val Loss: {val_metrics['loss']:.4f} | Word Acc: {val_metrics['word_accuracy']:.2%}")
    
    if val_metrics['word_accuracy'] > best_val_accuracy:
        best_val_accuracy = val_metrics['word_accuracy']
        torch.save(manual_model.state_dict(), 'manual_best_model.pt')

# Load best validation model
manual_model.load_state_dict(torch.load('manual_best_model.pt'))

# Evaluate on test set
test_accuracy, test_predictions, confusion_mat = evaluate_test_set(
    manual_model, test_data, latin_vocab, devanagari_vocab
)

print(f"\nTest Set Word Accuracy: {test_accuracy:.2%}")

# Create a colorful comparison table
from tabulate import tabulate

sample = test_predictions.sample(10, random_state=42)
table = []
for _, row in sample.iterrows():
    status = "✓" if row['correct'] else "✗"
    color = '\033[92m' if row['correct'] else '\033[91m'
    table.append([
        row['source'],
        row['target'],
        row['prediction'],
        f"{color}{status}\033[0m"
        ])

print("\n\033[1mSample Predictions:\033[0m")
print(tabulate(table, 
               headers=["Source", "Target", "Predicted", "Correct"], 
               tablefmt="grid"))

# Error analysis
errors = test_predictions[~test_predictions['correct']]
correct = test_predictions[test_predictions['correct']]

print("\n\033[1mError Analysis:\033[0m")
print(f"- Total errors: {len(errors)}/{len(test_predictions)} ({len(errors)/len(test_predictions):.2%})")
print(f"- Average length of correct words: {correct['target'].str.len().mean():.1f}")
print(f"- Average length of incorrect words: {errors['target'].str.len().mean():.1f}")

# Character-level error breakdown
vowels = set('अआइईउऊऋएऐओऔ')
consonants = set('कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')

def count_errors(row):
    target_chars = set(row['target'])
    pred_chars = set(row['prediction'])
    diff = target_chars - pred_chars
    return {
        'vowel_errors': sum(1 for c in diff if c in vowels),
        'consonant_errors': sum(1 for c in diff if c in consonants)
    }

error_counts = errors.apply(count_errors, axis=1, result_type='expand')
print(f"- Consonant errors: {error_counts['consonant_errors'].sum()}")
print(f"- Vowel errors: {error_counts['vowel_errors'].sum()}")