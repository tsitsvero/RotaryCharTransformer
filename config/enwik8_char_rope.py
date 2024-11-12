import math

# Configuration for the modified model
out_dir = 'out'
eval_interval = 500
eval_iters = 200
log_interval = 200

always_save_checkpoint = True
wandb_log = True
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-rope'

dataset = 'enwik8'
gradient_accumulation_steps = 1 #8
batch_size = 64  # Reduced batch size for stability
block_size = 256  # Reduced context length

# Model parameters
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.1  # Removed dropout initially for debugging
bias = False

# Optimization parameters
learning_rate = 2e-4
max_iters = 25000
lr_decay_iters = 25000
min_lr = 2e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100

# Model type and initialization
init_from = 'scratch'
model_type = 'rope'
use_rational=True  # Explicitly disable Rational activation
use_stiefel = False

# System parameters
device = 'cuda'
dtype = 'float16'  # Changed to bfloat16 for better stability
compile = False

# Fixed vocab size for enwik8
vocab_size = 256  # For byte-level encoding

# Add use_rotary parameter to config
use_rotary = True  # Enable RoPE

# Let's see the contents of this file