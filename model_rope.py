import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig
import inspect


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

        # Token embeddings
        tok_emb = self.transformer.wte(idx)  # shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
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
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_stiefel=True):
        """Configure optimizers with option to use Stiefel optimizer for attention Q,K weights"""
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        if use_stiefel:
            # Separate parameters into Stiefel (Q,K weights) and Euclidean groups
            stiefel_params = []
            euclidean_decay_params = []
            euclidean_nodecay_params = []

            for n, p in param_dict.items():
                # Q,K weights from attention should be on Stiefel manifold
                if any(x in n for x in ['q.weight', 'k.weight']):
                    torch.nn.init.orthogonal_(p)  # Initialize Q,K weights to be orthogonal
                    stiefel_params.append(p)
                # Regular weight matrices get weight decay
                elif p.dim() >= 2:
                    euclidean_decay_params.append(p)
                # Biases and LayerNorm parameters don't get weight decay
                else:
                    euclidean_nodecay_params.append(p)

            # Create parameter groups for Euclidean Adam
            euclidean_groups = [
                {'params': euclidean_decay_params, 'weight_decay': weight_decay},
                {'params': euclidean_nodecay_params, 'weight_decay': 0.0}
            ]

            # Create optimizers
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            
            euclidean_optimizer = torch.optim.AdamW(euclidean_groups, lr=learning_rate, betas=betas, **extra_args)
            stiefel_optimizer = StiefelAdam([{'params': stiefel_params}], lr=learning_rate, betas=betas)
            
            # Combine optimizers
            optimizer = CombinedOptimizer(euclidean_optimizer, stiefel_optimizer)
            
            print(f"Using Stiefel optimizer for {len(stiefel_params)} Q,K weight matrices")
            print(f"Euclidean optimizer: {len(euclidean_decay_params)} decay params, {len(euclidean_nodecay_params)} no-decay params")
            print(f"Using fused AdamW for Euclidean params: {use_fused}")

        else:
            # Default AdamW optimization for all parameters
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            
            print(f"Using standard AdamW for all parameters")
            print(f"num decayed parameter tensors: {len(decay_params)}")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}")
            print(f"using fused AdamW: {use_fused}")

        return optimizer



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
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

        # Precompute rotary embeddings
        self.rotary_emb = RotaryEmbedding(dim=config.n_embd // config.n_head)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, n_head, T, head_dim)

        # Apply rotary embeddings to q and k
        q, k = self.rotary_emb(q, k)  # Correcting this line

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k):
        t = q.size(-2)
        freqs = torch.einsum("i,j->ij", torch.arange(t, device=q.device).float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)  # Fix the call for 'k'
        return q, k


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)  # Use standard GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x