import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig
import inspect
from dataclasses import dataclass
from rational import Rational

 
# Import StiefelAdam
from StiefelOptimizers import StiefelAdam, CombinedOptimizer

def apply_rotary_pos_emb(q, cos, sin):
    # Apply rotary position embedding to query and key
    q_cos = q * cos
    q_sin = q * sin
    q_rotated = q_cos + rotate_half(q_sin)
    return q_rotated


def rotate_half(x):
    # Helper function to apply rotation
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

class GPTWithRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert idx.max() < self.config.vocab_size, f"Input contains token {idx.max()} which is >= vocab size {self.config.vocab_size}"
        
        # Token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx.long())  # Ensure long dtype for embeddings
        
        # Forward pass through transformer blocks
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Calculate loss if targets are provided
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.long().view(-1), ignore_index=-1)
        else:
            # For inference, only compute logits for the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    #     # Start with all of the candidate parameters
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     # Filter out those that do not require grad
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    #     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #     optim_groups = [
    #         {'params': decay_params, 'weight_decay': weight_decay},
    #         {'params': nodecay_params, 'weight_decay': 0.0}
    #     ]
    #     num_decay_params = sum(p.numel() for p in decay_params)
    #     num_nodecay_params = sum(p.numel() for p in nodecay_params)
    #     print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    #     print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    #     # Create AdamW optimizer
    #     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #     use_fused = fused_available and device_type == 'cuda'
    #     extra_args = dict(fused=True) if use_fused else dict()
    #     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    #     print(f"using fused AdamW: {use_fused}")

    #     return optimizer
    
    # def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_stiefel=True):
    #     """Configure optimizers with option to use Stiefel optimizer for attention Q,K weights"""
    #     # start with all of the candidate parameters
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     # filter out those that do not require grad
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    #     if use_stiefel:
    #         # Separate parameters into Stiefel (Q,K weights) and Euclidean groups
    #         stiefel_params = []
    #         euclidean_decay_params = []
    #         euclidean_nodecay_params = []

    #         for n, p in param_dict.items():
    #             # Q,K weights from attention should be on Stiefel manifold
    #             if any(x in n for x in ['q.weight', 'k.weight']):
    #                 torch.nn.init.orthogonal_(p)  # Initialize Q,K weights to be orthogonal
    #                 stiefel_params.append(p)
    #             # Regular weight matrices get weight decay
    #             elif p.dim() >= 2:
    #                 euclidean_decay_params.append(p)
    #             # Biases and LayerNorm parameters don't get weight decay
    #             else:
    #                 euclidean_nodecay_params.append(p)

    #         # Create parameter groups for Euclidean Adam
    #         euclidean_groups = [
    #             {'params': euclidean_decay_params, 'weight_decay': weight_decay},
    #             {'params': euclidean_nodecay_params, 'weight_decay': 0.0}
    #         ]

    #         # Create optimizers
    #         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #         use_fused = fused_available and device_type == 'cuda'
    #         extra_args = dict(fused=True) if use_fused else dict()
            
    #         euclidean_optimizer = torch.optim.AdamW(euclidean_groups, lr=learning_rate, betas=betas, **extra_args)
    #         stiefel_optimizer = StiefelAdam([{'params': stiefel_params}], lr=learning_rate, betas=betas)
            
    #         # Combine optimizers
    #         optimizer = CombinedOptimizer(euclidean_optimizer, stiefel_optimizer)
            
    #         print(f"Using Stiefel optimizer for {len(stiefel_params)} Q,K weight matrices")
    #         print(f"Euclidean optimizer: {len(euclidean_decay_params)} decay params, {len(euclidean_nodecay_params)} no-decay params")
    #         print(f"Using fused AdamW for Euclidean params: {use_fused}")

    #     else:
    #         # Default AdamW optimization for all parameters
    #         decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    #         nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    #         optim_groups = [
    #             {'params': decay_params, 'weight_decay': weight_decay},
    #             {'params': nodecay_params, 'weight_decay': 0.0}
    #         ]
            
    #         fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #         use_fused = fused_available and device_type == 'cuda'
    #         extra_args = dict(fused=True) if use_fused else dict()
    #         optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            
    #         print(f"Using standard AdamW for all parameters")
    #         print(f"num decayed parameter tensors: {len(decay_params)}")
    #         print(f"num non-decayed parameter tensors: {len(nodecay_params)}")
    #         print(f"using fused AdamW: {use_fused}")

    #     return optimizer



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.rotary_emb = RotaryEmbedding(dim=config.n_embd // config.n_head)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.k(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.v(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply rotary embeddings to q and k
        q, k = self.rotary_emb(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, q, k):
        seq_len = q.shape[-2]
        self._update_cos_sin_tables(q, seq_len)
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached)
        )


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # Choose activation function based on config
        # Report if using rational activation
        print(f"Using {'Rational' if config.use_rational else 'GELU'} activation function")
        self.act = Rational() if config.use_rational else nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x