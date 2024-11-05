import torch
from torch.optim import Optimizer
import torch.nn.functional as F
import math

def orthogonality_penalty(W):
    WtW = W @ W.t()
    I = torch.eye(WtW.size(0), device=W.device)
    return torch.norm(WtW - I, p='fro')

class HeadwiseStiefelAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, ortho_lambda=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.ortho_lambda = ortho_lambda
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

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]

                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state['step'] += 1
                    state_steps.append(state['step'])

            # Update parameters for this group
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                
                # Scale gradients based on manifold curvature
                grad_norm = torch.norm(grad)
                manifold_scale = torch.sqrt(param.size(0))
                grad = grad * (manifold_scale / (grad_norm + group['eps']))

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = group['lr']

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Apply bias correction and weight decay
                step_size = group['lr'] / bias_correction1
                
                # Weight decay term (AdamW-style)
                if group['weight_decay'] != 0:
                    param.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Compute the update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = exp_avg / denom
                
                # Project update onto tangent space of Stiefel manifold
                A = update @ param.t()
                skew = (A - A.t()) / 2
                update = update - param @ A + param @ skew
                
                # Update parameter
                param.add_(update, alpha=-step_size)
                
                # Retraction step using QR decomposition (more stable than Cayley)
                q, r = torch.linalg.qr(param)
                param.copy_(q)

        return loss