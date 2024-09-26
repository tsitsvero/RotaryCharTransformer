out_dir = 'out-enwik8-char'
eval_interval = 500
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
wandb_log = False
wandb_project = 'enwik8-char'
wandb_run_name = 'gpt2-enwik8-char-baseline'

dataset = 'enwik8'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 12
n_head = 8
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 1e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0
decay_lr = True
warmup_iters = 100
init_from = 'scratch'

device = 'cuda'
dtype = 'float16'
compile = False