import os
import pickle
import numpy as np

# Define the data directory where the files will be stored
data_dir = 'data/enwik8'

# Safely read the text files, ensuring they exist
def read_file_safe(filename):
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filename} not found in {data_dir}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# Load the dataset splits
train_data = read_file_safe('train.txt')
val_data = read_file_safe('valid.txt')
test_data = read_file_safe('test.txt')

# Extract unique characters from the training data
chars = sorted(set(train_data))
vocab_size = len(chars)
print(f"Vocabulary size (unique characters): {vocab_size}")

# Create character-to-integer (stoi) and integer-to-character (itos) mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Save the mappings as a metadata file for future use
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
meta_filepath = os.path.join(data_dir, 'meta.pkl')
with open(meta_filepath, 'wb') as f:
    pickle.dump(meta, f)
print(f"Saved metadata to {meta_filepath}")

# Function to encode data into integer token IDs
def encode_data(text):
    return [stoi[ch] for ch in text if ch in stoi]

# Encode the training, validation, and test data
train_ids = np.array(encode_data(train_data), dtype=np.uint16)
val_ids = np.array(encode_data(val_data), dtype=np.uint16)
test_ids = np.array(encode_data(test_data), dtype=np.uint16)

# Save the encoded data to binary files for efficient loading during training
train_bin_filepath = os.path.join(data_dir, 'train.bin')
val_bin_filepath = os.path.join(data_dir, 'val.bin')
test_bin_filepath = os.path.join(data_dir, 'test.bin')

train_ids.tofile(train_bin_filepath)
val_ids.tofile(val_bin_filepath)
test_ids.tofile(test_bin_filepath)

print(f"Data preparation complete. Files saved at: \n- {train_bin_filepath}\n- {val_bin_filepath}\n- {test_bin_filepath}")