import os
import torch
from model import GPT, GPTConfig

def encode_string(s):
    """Convert string to list of byte values"""
    return [ord(c) for c in s]

def decode_bytes(b):
    """Convert list of byte values back to string"""
    return bytes(b).decode('utf-8', errors='replace')

def main():
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model configuration
    config = GPTConfig(
        block_size=256,        # Smaller context for demonstration
        vocab_size=256,        # Use 256 for byte-level encoding
        n_layer=12,            # 12 transformer blocks
        n_head=12,            # 12 attention heads
        n_embd=768,           # 768 embedding dimension
        use_rational=True,     # Use Rational activation function
    )

    # Create model
    model = GPT(config)
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Load the checkpoint
    ckpt_path = 'out/ckpt.pt'  # Default checkpoint path
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Load the model state dict
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("Warning: No checkpoint found. Using random initialization")

    # Input prompt
    prompt = "historia de "
    
    # Convert prompt to byte values and create tensor
    input_bytes = encode_string(prompt)
    input_tensor = torch.tensor(input_bytes, dtype=torch.long)[None, ...].to(device)
    
    # Generate text
    print("\nPrompt:", prompt)
    print("\nGenerating with temperature = 0.8:")
    
    # Generation parameters
    max_new_tokens = 200  
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