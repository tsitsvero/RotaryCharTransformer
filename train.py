#modified train.py
import argparse
import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect
from collections import defaultdict

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

# Add these imports at the top
from torch.utils.data import Dataset, DataLoader
import threading
import queue
import io

class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        # Read data once during initialization
        with open(data_path, 'rb') as f:
            self.data = f.read()
        self.block_size = block_size
        
    def __len__(self):
        return len(self.data) - self.block_size
        
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor([int(b) for b in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([int(b) for b in chunk[1:]], dtype=torch.long)
        return x, y

class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        try:
            self.next_x, self.next_y = next(self.loader)
        except StopIteration:
            self.next_x = None
            self.next_y = None
            return

        with torch.cuda.stream(self.stream):
            self.next_x = self.next_x.to(self.device, non_blocking=True)
            self.next_y = self.next_y.to(self.device, non_blocking=True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        x = self.next_x
        y = self.next_y
        self.preload()
        return x, y

class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        
    def log(self, operation, duration):
        self.times[operation] += duration
        self.counts[operation] += 1
    
    def get_average(self, operation):
        count = self.counts[operation]
        if count == 0:
            return 0.0
        return self.times[operation] / count
    
    def print_stats(self):
        print("\nTiming Statistics:")
        for op in self.times:
            avg_time = self.get_average(op)
            total_time = self.times[op]
            count = self.counts[op]
            print(f"{op:20s}: {avg_time*1000:8.2f}ms (avg) | {total_time:8.2f}s (total) | {count:6d} calls")

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

# Move timer initialization before get_batch definition
timer = Timer()

def get_batch(split, data_dir, config, device, device_type):
    """Get a batch using the prefetcher"""
    if not hasattr(get_batch, 'prefetcher'):
        # Initialize dataset and prefetcher on first call
        data_path = os.path.join(data_dir, f'{split}.txt')
        dataset = TextDataset(data_path, config['block_size'])
        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=4,
            pin_memory=True
        )
        get_batch.prefetcher = DataPrefetcher(loader, device)
    
    # Get next batch
    x, y = get_batch.prefetcher.next()
    if x is None:  # Reset if we reached the end
        get_batch.prefetcher = None
        return get_batch(split, data_dir, config, device, device_type)
    
    return x, y

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

    # Initialize timer here
    timer = Timer()

    print_dataset_sample(data_dir)

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
        # model = BaselineGPT(gptconf)
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
            if any(x in n for x in ['7.attn.q.weight', '7.attn.k.weight']): #['0.attn.q.weight']): 
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
                X, Y = get_batch(split, data_dir, config, device, device_type)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        """Implements cosine learning rate schedule with linear warmup"""
        # Linear warmup for warmup_iters steps
        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        
        # If we're past max_iters, return min learning rate
        if it > config['max_iters']:
            return config['min_lr']
        
        # Otherwise, use cosine decay schedule
        decay_ratio = (it - config['warmup_iters']) / (config['max_iters'] - config['warmup_iters'])
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        
        # Interpolate between max and min learning rate
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    X, Y = get_batch('train', data_dir, config, device, device_type)
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

    timer = Timer()

    # Initialize automatic mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' and config['dtype'] == 'float16' else None
    
    # Create gradient accumulation buffer
    grad_acc_steps = config['gradient_accumulation_steps']
    
    with tqdm(total=config['max_iters'], desc="Training Progress") as pbar:
        while iter_num < config['max_iters']:
            # Update learning rate according to schedule
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Time the full step
            step_start = time.time()
            
            # Zero gradients at the start of accumulation
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            
            # Accumulate gradients
            for micro_step in range(grad_acc_steps):
                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    logits, loss = model(X, Y)
                    loss = loss / grad_acc_steps
                
                # Backward pass with gradient scaling
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                total_loss += loss.item()
                
                # Prefetch next batch while computing gradients
                if micro_step < grad_acc_steps - 1:
                    X, Y = get_batch('train', data_dir, config, device, device_type)
            
            # Gradient clipping
            if config['grad_clip'] != 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            # Optimizer step with gradient scaling
            def closure():
                return loss

            if scaler is not None:
                scaler.step(optimizer, closure)
                scaler.update()
            else:
                if config['grad_clip'] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step(closure)
            
            # Get next batch for next iteration
            X, Y = get_batch('train', data_dir, config, device, device_type)
            
            opt_time = time.time() - step_start
            timer.log('total_step', opt_time)

            # Print timing stats periodically
            if iter_num % config['log_interval'] == 0 and master_process:
                # timer.print_stats()  # Comment out timing stats
                
                # Also log to wandb
                wandb.log({
                    'time/get_batch': timer.get_average('get_batch'),
                    'time/optimizer_step': timer.get_average('optimizer_step'),
                    'time/total_step': timer.get_average('total_step')
                })

            if iter_num % config['eval_interval'] == 0 and master_process:
                losses = estimate_loss()
                
                # Calculate BPC (bits per character) from losses
                train_bpc = losses['train'] / math.log(2)
                valid_bpc = losses['valid'] / math.log(2)
                
                # Print losses and BPC
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f} ({train_bpc:.3f} bpc), "
                      f"valid loss {losses['valid']:.4f} ({valid_bpc:.3f} bpc)")
                
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
                    'train/bpc': train_bpc,
                    'valid/loss': losses['valid'],
                    'valid/bpc': valid_bpc,
                    'lr': lr,
                    'ortho/stiefel_mean': ortho_metrics['stiefel_mean'],
                    'ortho/stiefel_rel_mean': ortho_metrics['stiefel_rel_mean'],
                    'ortho/other_mean': ortho_metrics['other_mean'],
                    'ortho/other_rel_mean': ortho_metrics['other_rel_mean'],
                    'ortho/stiefel_max': ortho_metrics['stiefel_max'][1],
                    'ortho/stiefel_max_rel': ortho_metrics['stiefel_max'][2],
                })
                
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}")
                
                # Log metrics to wandb
                wandb.log({
                    'iter': iter_num,
                    'train/loss': losses['train'],
                    'valid/loss': losses['valid'],
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

            if iter_num % config['log_interval'] == 1 and master_process:
                lossf = total_loss * config['gradient_accumulation_steps']
                print(f"Iter {iter_num}: loss {lossf:.4f}")
                
                # Generate and print sequence
                generated_text = generate_sequence(raw_model, device)
                print("\nGenerated sequence starting with 'hello':")
                print("=" * 80)
                print(generated_text)
                print("=" * 80)
                
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
            error = torch.norm(WtW - I, p='fro').item()
            rel_error = error / torch.norm(I, p='fro').item()
            
            if any(x in name for x in ['3.attn.q.weight']):
                stiefel_errors.append((name, error, rel_error))
            else:
                other_attn_errors.append((name, error, rel_error))
    
    return {
        'stiefel_mean': np.mean([e[1] for e in stiefel_errors]) if stiefel_errors else 0.0,
        'stiefel_rel_mean': np.mean([e[2] for e in stiefel_errors]) if stiefel_errors else 0.0,
        'other_mean': np.mean([e[1] for e in other_attn_errors]) if other_attn_errors else 0.0,
        'other_rel_mean': np.mean([e[2] for e in other_attn_errors]) if other_attn_errors else 0.0,
        'stiefel_max': max(stiefel_errors, key=lambda x: x[1]) if stiefel_errors else ('none', 0.0, 0.0),
        'other_max': max(other_attn_errors, key=lambda x: x[1]) if other_attn_errors else ('none', 0.0, 0.0)
    }

# Add this function after get_batch
def print_sample(model, x, y, config):
    """Print a random sample from the batch with its prediction"""
    # Select random sample from batch
    idx = torch.randint(0, x.shape[0], (1,)).item()
    sample_x = x[idx]
    sample_y = y[idx]
    
    # Get model prediction
    with torch.no_grad():
        logits, _ = model(sample_x.unsqueeze(0))
        probs = F.softmax(logits[0], dim=-1)
        pred = torch.argmax(probs, dim=-1)
    
    # Convert to string directly
    def bytes_to_string(tensor):
        bytes_data = tensor.cpu().numpy().astype(np.uint8).tobytes()
        return bytes_data.decode('utf-8', errors='replace')
    
    input_str = bytes_to_string(sample_x[:20])
    target_str = bytes_to_string(sample_y[:20])
    pred_str = bytes_to_string(pred[:20])
    
    print("\nRandom sample:")
    print(f"Input  (20 chars): {input_str}")
    print(f"Target (20 chars): {target_str}")
    print(f"Pred   (20 chars): {pred_str}")

# Add this function after print_sample
@torch.no_grad()
def generate_sequence(model, device, length=100, temperature=1.0):
    """Generate a sequence starting with 'hello'"""
    # Convert 'hello' to byte encoding
    context = torch.tensor([ord(c) for c in 'hello'], dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    for _ in range(length):
        # Get logits from model
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated.append(next_token.item())
        
        # Update context
        context = torch.cat((context, next_token), dim=1)
    
    # Convert bytes to string, including the 'hello' prefix
    try:
        result = 'hello' + bytes(generated).decode('utf-8', errors='replace')
    except:
        result = 'hello' + ''.join(chr(b) for b in generated)
    
    return result

if __name__ == '__main__':
    main()