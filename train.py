#modified train.py
import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm

from model import GPTConfig
from model_baseline import BaselineGPT
from model_rope import GPTWithRoPE


# Import StiefelAdam
from StiefelOptimizers import StiefelAdam, CombinedOptimizer

# Import HeadwiseStiefelAdam
# from headwise_stiefel_optimizer import HeadwiseStiefelAdam

# Add this import at the top of the file
import wandb

# Add these imports at the top
import torch.nn.functional as F

def get_serializable_config(config):
    return {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, type(None))) and not k.startswith('__')}

def print_dataset_sample(data_dir, split='train', n_chars=1000):
    """Print first n_chars of the dataset"""
    data_path = os.path.join(data_dir, f'{split}.txt')
    
    # Read the file
    with open(data_path, 'rb') as f:
        data = f.read(n_chars)
    
    print(f"\nFirst {n_chars} characters of {split} dataset:")
    print("=" * 80)
    print(data.decode('utf-8', errors='replace'))
    print("=" * 80)

def print_sample(model, device, prompt="hello ", max_new_tokens=100):
    """Generate and print a sample from the model"""
    model.eval()
    
    # Convert prompt to bytes and create input tensor
    input_bytes = [ord(c) for c in prompt]
    input_tensor = torch.tensor(input_bytes, dtype=torch.long)[None, ...].to(device)
    
    with torch.no_grad():
        # Generate with temperature = 0.8
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=40
        )
    
    # Convert output back to text
    generated_text = bytes(output_ids[0].cpu().tolist()).decode('utf-8', errors='replace')
    print("\nSample generation:")
    print("=" * 40)
    print(generated_text)
    print("=" * 40)
    
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args = parser.parse_args()

    config_file = args.config
    config = {}
    with open(config_file, 'r') as f:
        exec(f.read(), {}, config)

    config = {k: v for k, v in config.items() if not k.startswith('__')}

    if 'out_dir' not in config:
        print("Error: 'out_dir' not specified in the configuration file.")
        return

    if int(os.environ.get('RANK', -1)) == -1:
        os.makedirs(config['out_dir'], exist_ok=True)
        print(f"Output directory: {config['out_dir']}")

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1
        device = config['device']

    tokens_per_iter = (config['gradient_accumulation_steps'] * ddp_world_size *
                       config['batch_size'] * config['block_size'])
    print(f"Tokens per iteration will be: {tokens_per_iter:,}")

    torch.manual_seed(1337 + int(time.time()))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    data_dir = os.path.join('data', config['dataset'])

    print_dataset_sample(data_dir)

    # Add this after data_dir initialization but before the training loop
    def load_data(data_dir):
        """Load train and validation data into memory more efficiently"""
        data_dict = {}
        for split in ['train', 'valid']:
            data_path = os.path.join(data_dir, f'{split}.txt')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Use memory mapping for large files
            data = np.memmap(data_path, dtype=np.uint8, mode='r')
            
            # Convert to tensor more efficiently
            data_tensor = torch.from_numpy(data.astype(np.int64))
            if device_type == 'cuda':
                data_tensor = data_tensor.pin_memory().to(device, non_blocking=True)
            else:
                data_tensor = data_tensor.to(device)
                
            data_dict[split] = data_tensor
            
        return data_dict

    # Load all data into memory
    print("Loading data into memory...")
    data_tensors = load_data(data_dir)
    print("Data loaded successfully!")

    # Modify get_batch to use the preloaded data
    def get_batch(split):
        """Get a random batch more efficiently"""
        data = data_tensors[split]
        ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],), device=device)
        
        # Use advanced indexing for better performance
        x = torch.stack([data[i:i+config['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+1+config['block_size']] for i in ix])
        
        return x, y

    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # vocab_size = meta['vocab_size']
    else:
        meta = {'vocab_size': 256}  # Default for byte encoding

    config['vocab_size'] = 256  # Force vocab size to 256 for byte-level encoding

    gpt_config_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']
    gpt_config = {k: v for k, v in config.items() if k in gpt_config_keys}
    gptconf = GPTConfig(**gpt_config)

    if config.get('model_type') == 'rope':
        model = GPTWithRoPE(gptconf)
        print("Using GPTWithRoPE model.")
    else:
        model = BaselineGPT(gptconf)
        print("Using BaselineGPT model.")

    model.to(device)

    use_stiefel = config.get('use_stiefel', True)

    if not use_stiefel:

        # Initialize optimizer outside of the model
        decay_params = [p for p in model.parameters() if p.dim() >= 2]
        no_decay_params = [p for p in model.parameters() if p.dim() < 2]

        optimizer = optim.AdamW([
            {'params': decay_params, 'weight_decay': config['weight_decay']},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

        if config['dtype'] == 'float16':
            scaler = torch.cuda.amp.GradScaler()
        else:
            scaler = None

    else:
        # Separate parameters into Stiefel (Q,K weights) and Euclidean groups
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        stiefel_params = []
        euclidean_decay_params = []
        euclidean_nodecay_params = []

        for n, p in param_dict.items():
            # Apply Stiefel to Q,K weights in attention layers
            if any(x in n for x in ['3.attn.q.weight']):#['attn.q.weight', 'attn.k.weight']): #['0.attn.q.weight']): 
                # Initialize orthogonally and add to Stiefel params
                # torch.nn.init.orthogonal_(p)
                stiefel_params.append(p)
                print(f"Added Stiefel parameter: {n}, shape: {p.shape}")
            
            # All other parameters use regular optimization
            elif p.dim() >= 2:
                euclidean_decay_params.append(p)
            else:
                euclidean_nodecay_params.append(p)

        # Create parameter groups for Euclidean Adam
        euclidean_groups = [
            {'params': euclidean_decay_params, 'weight_decay': config['weight_decay']},
            {'params': euclidean_nodecay_params, 'weight_decay': 0.0}
        ]

        # Create optimizers
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        euclidean_optimizer = torch.optim.AdamW(euclidean_groups, lr=config['learning_rate'], 
                                               betas=(config['beta1'], config['beta2']), **extra_args)
        
        # Use StiefelAdam for Q,K weights
        stiefel_optimizer = StiefelAdam([{'params': stiefel_params}], 
                                      lr=config['learning_rate'],
                                      betas=(config['beta1'], config['beta2']))
        
        # Combine optimizers
        optimizer = CombinedOptimizer(euclidean_optimizer, stiefel_optimizer)
        
        print(f"Using Stiefel optimizer for {len(stiefel_params)} Q,K weight matrices")
        print(f"Euclidean optimizer: {len(euclidean_decay_params)} decay params, {len(euclidean_nodecay_params)} no-decay params")
        print(f"Using fused AdamW for Euclidean params: {use_fused}")

        if config['dtype'] == 'float16':
            scaler = torch.amp.GradScaler('cuda')
        else:
            scaler = None





    iter_num = 0
    best_val_loss = 1e9

    if config.get('init_from', 'scratch') == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from iteration {iter_num}, best val loss {best_val_loss}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params/1e6:.2f}M")

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(config['eval_iters'])
            for k in range(config['eval_iters']):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        if it > config['lr_decay_iters']:
            return config['min_lr']
        decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    X, Y = get_batch('train')
    running_mfu = -1.0
    t0 = time.time()

    local_iter_num = 0
    raw_model = model.module if ddp else model

    # Initialize wandb
    if master_process:
        # Filter out non-serializable config values
        wandb_config = {
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'block_size': config['block_size'],
            'max_iters': config['max_iters'],
            'weight_decay': config['weight_decay'],
            'beta1': config['beta1'],
            'beta2': config['beta2'],
            'grad_clip': config['grad_clip'],
            'decay_lr': config['decay_lr'],
            'min_lr': config['min_lr'],
            'warmup_iters': config['warmup_iters'],
            'lr_decay_iters': config['lr_decay_iters'],
            'eval_interval': config['eval_interval'],
            'log_interval': config['log_interval'],
            'dataset': config['dataset'],
            'dtype': config['dtype'],
            'model_type': config.get('model_type', 'baseline'),
            'n_layer': config['n_layer'],
            'n_head': config['n_head'],
            'n_embd': config['n_embd'],
            'dropout': config['dropout'],
            'bias': config['bias'],
            'vocab_size': config['vocab_size'],
            # Add custom tracking fields
            'use_stiefel': use_stiefel,
            'use_rational': config.get('use_rational', False),
            'total_params': total_params,
            'device_type': device_type,
            'gradient_accumulation_steps': config['gradient_accumulation_steps'],
            'tokens_per_iter': tokens_per_iter,
        }
        
        wandb.init(
            project="transformer-stiefel",
            config=wandb_config,
            name=f"{config.get('model_type', 'baseline')}_stiefel_{use_stiefel}"
        )

    with tqdm(total=config['max_iters'], desc="Training Progress") as pbar:
        while iter_num < config['max_iters']:
            lr = config['learning_rate'] if not config['decay_lr'] else get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Zero gradients once for all micro-batches
            optimizer.zero_grad(set_to_none=True)
            
            # Pre-fetch batches for all micro-steps
            batches = [(get_batch('train')) for _ in range(config['gradient_accumulation_steps'])]
            
            total_loss = 0.0
            
            # Accumulate gradients
            for micro_step, (X, Y) in enumerate(batches):
                if ddp:
                    model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
                
                # Forward pass with automatic mixed precision
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / config['gradient_accumulation_steps']
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                total_loss += loss.item()

            # Optimization step
            if scaler is not None:
                scaler.unscale_(optimizer)
                if config['grad_clip'] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                if use_stiefel:
                    scaler.step(optimizer, lambda: total_loss)
                else:
                    scaler.step(optimizer)
                scaler.update()
            else:
                if config['grad_clip'] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                if use_stiefel:
                    optimizer.step(lambda: total_loss)
                else:
                    optimizer.step()

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss()
                
                # Generate and print sample text
                print_sample(raw_model, device)
                
                # Print losses and other metrics
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
                
                # Check orthogonality for all attention matrices
                ortho_metrics = compute_orthogonality_error(raw_model)
                print(f"Orthogonality errors:")
                print(f"Stiefel matrices - mean: {ortho_metrics['stiefel_mean']:.6f}, "
                      f"max: {ortho_metrics['stiefel_max'][1]:.6f} ({ortho_metrics['stiefel_max'][0]})")
                print(f"Other matrices  - mean: {ortho_metrics['other_mean']:.6f}, "
                      f"max: {ortho_metrics['other_max'][1]:.6f} ({ortho_metrics['other_max'][0]})")
                
                # Log metrics to wandb
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['valid'],
                    'lr': lr,
                    'ortho/stiefel_mean': ortho_metrics['stiefel_mean'],
                    'ortho/other_mean': ortho_metrics['other_mean'],
                    'ortho/stiefel_max': ortho_metrics['stiefel_max'][1],
                    'ortho/other_max': ortho_metrics['other_max'][1]
                })
                
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}")
                
                # Log metrics to wandb
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'val/loss': losses['valid'],
                    'lr': lr,
                })
                
                if losses['valid'] < best_val_loss or config['always_save_checkpoint']:
                    best_val_loss = losses['valid']
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': get_serializable_config(config),
                    }
                    checkpoint_path = os.path.join(config['out_dir'], 'ckpt.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
                    # Log best val loss to wandb
                    wandb.log({'best_val_loss': best_val_loss})

            iter_num += 1
            local_iter_num += 1
            pbar.update(1)

            if iter_num % config['log_interval'] == 0 and master_process:
                lossf = total_loss * config['gradient_accumulation_steps']
                print(f"Iter {iter_num}: loss {lossf:.4f}")
                
                # Log training metrics
                wandb.log({
                    'iter': iter_num,
                    'train/batch_loss': lossf,
                    'lr': lr,
                })

    if master_process:
        wandb.finish()

    if ddp:
        destroy_process_group()

@torch.no_grad()
def compute_orthogonality_error(model):
    """
    Computes how far matrices are from being orthogonal.
    Returns separate mean deviations for Stiefel and non-Stiefel matrices.
    """
    stiefel_errors = []
    other_attn_errors = []
    
    for name, param in model.named_parameters():
        if 'attn' in name and '.weight' in name:
            # Compute orthogonality error for weight matrix
            W = param
            if W.shape[0] > W.shape[1]:  # Handle non-square matrices
                WtW = W.T @ W
                I = torch.eye(WtW.shape[0], device=WtW.device)
            else:
                WtW = W @ W.T
                I = torch.eye(WtW.shape[0], device=WtW.device)
            error = torch.norm(WtW - I, p='fro')
            
            # Separate Stiefel and non-Stiefel matrices
            if any(x in name for x in ['3.attn.q.weight']):
                stiefel_errors.append((name, error.item()))
            else:
                other_attn_errors.append((name, error.item()))
    
    # Calculate mean errors
    stiefel_mean = np.mean([e[1] for e in stiefel_errors]) if stiefel_errors else 0.0
    other_mean = np.mean([e[1] for e in other_attn_errors]) if other_attn_errors else 0.0
    
    # Get max errors with corresponding layer names
    stiefel_max = max(stiefel_errors, key=lambda x: x[1]) if stiefel_errors else ('none', 0.0)
    other_max = max(other_attn_errors, key=lambda x: x[1]) if other_attn_errors else ('none', 0.0)
    
    return {
        'stiefel_mean': stiefel_mean,
        'other_mean': other_mean,
        'stiefel_max': stiefel_max,
        'other_max': other_max
    }

if __name__ == '__main__':
    main()