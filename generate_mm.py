import os
import torch
import argparse
from model import GPT, GPTConfig
from model_rope import GPTWithRoPE
from torch.nn import functional as F

def encode_string(s):
    """Convert string to list of byte values"""
    return [ord(c) for c in s]

def decode_bytes(b):
    """Convert list of byte values back to string"""
    return bytes(b).decode('utf-8', errors='replace')

def get_next_token(logits, temperature=1.0, top_k=None):
    """Helper function to sample from the model's output distribution"""
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

class BaseGPTMixin:
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens using the model"""
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # get model predictions
            logits = self(idx_cond)
            # sample from the distribution
            next_token = get_next_token(logits, temperature, top_k)
            # append sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# Modify the model classes to include generation capability
class GPT(BaseGPTMixin, GPT):
    pass

class GPTWithRoPE(BaseGPTMixin, GPTWithRoPE):
    pass

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate text from a prompt')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt for generation')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of tokens to generate')
    args = parser.parse_args()

    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the checkpoint with weights_only=True for security
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Get configuration from checkpoint
    ckpt_config = checkpoint['config']
    
    # Force vocab_size to 256 for byte-level encoding
    ckpt_config['vocab_size'] = 256
    
    # Filter config to only include keys that GPTConfig accepts
    valid_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']
    model_config_kwargs = {k: ckpt_config[k] for k in valid_config_keys if k in ckpt_config}
    
    # Create model configuration
    config = GPTConfig(**model_config_kwargs)

    # Create appropriate model based on checkpoint type
    if ckpt_config.get('model_type') == 'rope':
        model = GPTWithRoPE(config)
        print("Using GPTWithRoPE model")
    else:
        model = GPT(config)
        print("Using GPT model")
        
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Load the model state
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Convert prompt to byte values and create tensor
    input_bytes = encode_string(args.prompt)
    input_tensor = torch.tensor(input_bytes, dtype=torch.long)[None, ...].to(device)
    
    # Generate text
    print("\nPrompt:", args.prompt)
    print("\nGenerating with temperature = 0.8:")
    
    # Generation parameters
    max_new_tokens = args.max_new_tokens
    temperature = 0.8    
    top_k = 40          
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode and print the generated text
    generated_text = decode_bytes(output_ids[0].cpu().tolist())
    print(generated_text)

    # Generate with different temperature
    print("\nGenerating with temperature = 1.2 (more random):")
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens,
            temperature=1.2,
            top_k=top_k
        )
    
    generated_text = decode_bytes(output_ids[0].cpu().tolist())
    print(generated_text)

    # Generate with lower temperature
    print("\nGenerating with temperature = 0.5 (more focused):")
    with torch.no_grad():
        output_ids = model.generate(
            input_tensor, 
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_k=top_k
        )
    
    generated_text = decode_bytes(output_ids[0].cpu().tolist())
    print(generated_text)

if __name__ == '__main__':
    main()