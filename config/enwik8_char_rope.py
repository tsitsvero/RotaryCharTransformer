import math

# Configuration for the enwik8 character-level model with RoPE
out_dir = 'out-enwik8-char-rope'
eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
wandb_log = True
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-rope'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # Increased context length for better modeling

# Model parameters
n_layer = 6
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# Optimization parameters
learning_rate = 6e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 6e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100

# Model type and initialization
init_from = 'scratch'
model_type = 'rope'
use_rational = False
use_stiefel = True

# System parameters
device = 'cuda'
dtype = 'float16'
compile = False


# Fixed vocab size for enwik8
vocab_size = 256  # For byte-level encoding