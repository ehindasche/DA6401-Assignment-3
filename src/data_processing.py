from libraries import *

# Load and preprocess data
def load_data(language='hi'):
    # Load the Dakshina dataset
    # Replace with actual paths to the dataset
    train_path = f'/kaggle/input/dakshina-dataset-seq2seq-for-transliteration/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
    dev_path = f'/kaggle/input/dakshina-dataset-seq2seq-for-transliteration/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
    test_path = f'/kaggle/input/dakshina-dataset-seq2seq-for-transliteration/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
    
    # Read TSV files with proper formatting
    train_data = pd.read_csv(train_path, sep='\t', names=['devanagari', 'latin', 'col3'])
    train_data = train_data.drop('col3', axis=1)
    dev_data = pd.read_csv(dev_path, sep='\t', names=['devanagari', 'latin', 'col3'])
    dev_data = dev_data.drop('col3', axis=1)
    test_data = pd.read_csv(test_path, sep='\t', names=['devanagari', 'latin', 'col3'])
    test_data = test_data.drop('col3', axis=1)

    # Convert all data to strings and strip whitespace
    train_data = train_data.applymap(lambda x: str(x).strip())
    dev_data = dev_data.applymap(lambda x: str(x).strip())
    test_data = test_data.applymap(lambda x: str(x).strip())
    
    return train_data, dev_data, test_data

# Create vocabulary
def create_vocab(data):
    latin_chars = set()
    devanagari_chars = set()
    
    for _, row in data.iterrows():
        # Ensure we're processing strings
        latin_word = str(row['latin'])
        devanagari_word = str(row['devanagari'])
        
        latin_chars.update(list(latin_word))
        devanagari_chars.update(list(devanagari_word))
    
    # Sort characters and build vocab with offset for special tokens
    sorted_latin_chars = sorted(latin_chars)
    sorted_devanagari_chars = sorted(devanagari_chars)

    latin_vocab = {char: idx + 4 for idx, char in enumerate(sorted_latin_chars)}
    devanagari_vocab = {char: idx + 4 for idx, char in enumerate(sorted_devanagari_chars)}
    
    # Add special tokens
    special_tokens = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    latin_vocab.update(special_tokens)
    devanagari_vocab.update(special_tokens)
    
    return latin_vocab, devanagari_vocab

# Dataset class
class TransliterationDataset(Dataset):
    # takes input as data and both vocabs
    def __init__(self, data, latin_vocab, devanagari_vocab):
        self.data = data
        self.latin_vocab = latin_vocab
        self.devanagari_vocab = devanagari_vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        latin = self.data.iloc[idx]['latin'] # takes away the 'latin' col
        devanagari = self.data.iloc[idx]['devanagari'] # takes away the 'devanagari' col        
        # Convert to indices
        latin_indices = [self.latin_vocab['<SOS>']] + \
                        [self.latin_vocab.get(c, self.latin_vocab['<UNK>']) for c in latin] + \
                        [self.latin_vocab['<EOS>']]
        # goes through the latin array and if found in latin_vocab, assigns the value. If not found assigns unknown
        # handles SOS & EOS seperately
        # output is an array of indices according to the vocabulary
                        
        devanagari_indices = [self.devanagari_vocab['<SOS>']] + \
                            [self.devanagari_vocab.get(c, self.devanagari_vocab['<UNK>']) for c in devanagari] + \
                            [self.devanagari_vocab['<EOS>']]
        # converts to torch tensor
        return torch.tensor(latin_indices, dtype=torch.long), torch.tensor(devanagari_indices, dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch):
    latin_batch, devanagari_batch = zip(*batch)
    
    # Pad sequences
    latin_padded = torch.nn.utils.rnn.pad_sequence(latin_batch, padding_value=0, batch_first=True)
    devanagari_padded = torch.nn.utils.rnn.pad_sequence(devanagari_batch, padding_value=0, batch_first=True)
    # returns an array of the provided data file with indices values instead of characters (strings)
    return latin_padded, devanagari_padded