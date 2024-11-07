import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Define GPTBlock with MultiheadAttention
class GPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout)
        self.drop = nn.Dropout(config.dropout)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed-forward layers
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # Apply layer normalization
        x_ln = self.ln_1(x)

        # Self-attention uses x_ln as query, key, and value
        attn_output, _ = self.attn(x_ln, x_ln, x_ln)
        x = x + self.drop(attn_output)

        # Feedforward block with residual connection
        x = x + self.drop(self.mlp(self.ln_2(x)))

        return x

# Define the main BaselineGPT model
class BaselineGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config

        # Transformer components
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),    # Token embedding
            'wpe': nn.Embedding(config.block_size, config.n_embd),    # Positional embedding
            'drop': nn.Dropout(config.dropout),                       # Dropout
            'h': nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),  # Stack of GPT blocks
            'ln_f': nn.LayerNorm(config.n_embd),                      # Final layer normalization
        })

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        # Initialize the weights for all components
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # Generate position indices and compute token and positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)
        tok_emb = self.transformer['wte'](idx)  # (b, t, n_embd)
        pos_emb = self.transformer['wpe'](pos)  # (1, t, n_embd)

        # Combine token and positional embeddings, then apply dropout
        x = self.transformer['drop'](tok_emb + pos_emb)

        # Pass through the stack of GPT blocks
        for block in self.transformer['h']:
            x = block(x)

        # Final layer normalization
        x = self.transformer['ln_f'](x)

        # Compute logits for language modeling
        logits = self.lm_head(x)

        # If targets are provided, compute loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss

        # Return logits if no targets are provided
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
