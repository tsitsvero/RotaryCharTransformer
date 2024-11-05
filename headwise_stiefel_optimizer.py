import torch
from torch.optim import Optimizer
import torch.nn.functional as F
import math

def orthogonality_penalty(W):
    WtW = W @ W.t()
    I = torch.eye(WtW.size(0), device=W.device)
    return torch.norm(WtW - I, p='fro')

class HeadwiseStiefelAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, ortho_lambda=0.01):
        self.ortho_lambda = ortho_lambda
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(HeadwiseStiefelAdam, self).__init__(params, defaults)
        
        # Initialize parameters more carefully
        for group in self.param_groups:
            for p in group['params']:
                # Get the shape of the parameter
                rows, cols = p.size()
                if rows > cols:
                    # If we have more rows than columns, pad with zeros
                    temp = torch.zeros(rows, rows, device=p.device)
                    temp[:, :cols] = p.data
                    q, r = torch.linalg.qr(temp)
                    p.data.copy_(q[:, :cols])
                else:
                    # If we have more columns than rows or equal dimensions
                    q, r = torch.linalg.qr(p.data.t())
                    p.data.copy_(q[:, :rows].t())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                # Add orthogonality penalty
                for group in self.param_groups:
                    for p in group['params']:
                        loss = loss + self.ortho_lambda * orthogonality_penalty(p)

        # Improved retraction using Cayley transform
        def cayley_retraction(X, G):
            n = X.size(0)
            I = torch.eye(n, device=X.device)
            A = G @ X.t() - X @ G.t()
            Q = I + A/2
            R = I - A/2
            Y = torch.linalg.solve(R, Q) @ X
            return Y

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            head_idx = group['head_idx']
            layer_name = group['layer_name']
            is_q = group['is_q']
            original_shape = group['original_shape']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Scale gradients based on manifold curvature
                grad_norm = torch.norm(p.grad)
                manifold_scale = torch.sqrt(p.size(0))  # Manifold dimension
                p.grad.mul_(manifold_scale / (grad_norm + 1e-8))

                # Get the original parameter from the model
                param_state = self.state[p]

                # Initialize state if needed
                if len(param_state) == 0:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)

                # Update step
                param_state['step'] += 1

                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                
                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                
                # Compute the update direction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg / bias_correction1 / denom
                
                # Project update onto tangent space of Stiefel manifold
                A = update @ p.t()
                skew = (A - A.t()) / 2
                update = update - p @ A + p @ skew
                
                # Update parameter while staying on Stiefel manifold
                p.add_(update, alpha=-lr)
                
                # Use Cayley retraction for final projection
                p_new = cayley_retraction(p, -lr * skew)
                p.copy_(p_new)

        return loss