import os
import torch
from model import GPT, GPTConfig

def main():
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model configuration
    config = GPTConfig(
        block_size=256,        # Smaller context for demonstration
        vocab_size=50304,      # Standard GPT-2 vocabulary
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
        # Fallback to pretrained GPT-2 if no checkpoint found
        try:
            model = GPT.from_pretrained('gpt2')
            print("Loaded pretrained GPT-2 weights")
        except:
            print("Warning: No checkpoint found and couldn't load pretrained weights")
            print("Using random initialization")

    # Example prompt
    prompt = "Once upon a time"
    
    # Initialize tokenizer
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    print("\nPrompt:", prompt)
    print("\nGenerating with temperature = 0.8:")
    
    # Generation parameters
    max_new_tokens = 100  # Increased length for more context
    temperature = 0.8    
    top_k = 40          
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(output_ids[0])
    print(generated_text)

    # Generate with different temperature
    print("\nGenerating with temperature = 1.2 (more random):")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=1.2,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output_ids[0])
    print(generated_text)

    # Generate with lower temperature
    print("\nGenerating with temperature = 0.5 (more focused):")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            top_k=top_k
        )
    
    generated_text = tokenizer.decode(output_ids[0])
    print(generated_text)

if __name__ == '__main__':
    main()