import torch
import numpy as np
import argparse
import pickle
import math
from model import GPTConfig, GPT
from model_rope import GPTWithRoPE

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            with torch.amp.autocast(device_type=device):
                logits, loss = model(x, y)
                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)  # Convert from nats to bits
    return avg_loss, bpc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['gpt', 'rope'], default='gpt', help='Model type: gpt or rope')
    parser.add_argument('--dataset', type=str, default='enwik8', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the checkpoint with weights_only=True for security
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Load the model configuration from the checkpoint
    ckpt_config = checkpoint['config']

    # Force vocab_size to 256 for byte-level encoding
    vocab_size = 256
    ckpt_config['vocab_size'] = vocab_size

    # Filter ckpt_config to only include keys that GPTConfig accepts
    valid_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']
    model_config_kwargs = {k: ckpt_config[k] for k in valid_config_keys if k in ckpt_config}

    # Model configuration
    model_config = GPTConfig(**model_config_kwargs)

    # Instantiate the model
    if args.model_type == 'rope' or ckpt_config.get('model_type') == 'rope':
        model = GPTWithRoPE(model_config)
        print("Using GPTWithRoPE model.")
    else:
        model = GPT(model_config)
        print("Using BaselineGPT model.")

    # Load the model state
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Prepare data loader
    block_size = ckpt_config['block_size']
    batch_size = ckpt_config.get('batch_size', 64)

    # Load validation data
    val_data = np.memmap(f'data/{args.dataset}/val.bin', dtype=np.uint8, mode='r')  # Changed to uint8
    val_data = torch.from_numpy(val_data.astype(np.int64))

    # Create sequences of block_size
    num_tokens = len(val_data) - 1
    x_tokens = val_data[:num_tokens]
    y_tokens = val_data[1:num_tokens+1]

    # Ensure that the number of tokens is a multiple of block_size
    num_batches = num_tokens // block_size
    x_tokens = x_tokens[:num_batches * block_size]
    y_tokens = y_tokens[:num_batches * block_size]

    # Reshape into batches
    x_batches = x_tokens.view(-1, block_size)
    y_batches = y_tokens.view(-1, block_size)

    val_dataset = torch.utils.data.TensorDataset(x_batches, y_batches)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Evaluate
    val_loss, bpc = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Bits per character (bpc): {bpc:.4f}")