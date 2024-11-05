import torch
from torch.optim import Optimizer
import torch.nn.functional as F

class HeadwiseStiefelAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(HeadwiseStiefelAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            head_idx = group['head_idx']
            is_q = group['is_q']
            original_shape = group['original_shape']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Reshape gradient to separate heads
                n_head = original_shape[0] // (original_shape[0] // p.shape[0])
                head_dim = original_shape[0] // n_head
                grad = p.grad.view(n_head, head_dim, -1)
                param = p.view(n_head, head_dim, -1)

                # Work with specific head
                head_grad = grad[head_idx]
                head_param = param[head_idx]

                # Get state for this head
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(head_param)
                    state['exp_avg_sq'] = torch.zeros_like(head_param)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Stiefel manifold projection
                A = head_grad @ head_param.t() - head_param @ head_grad.t()
                grad_stiefel = A @ head_param

                # Adam updates
                exp_avg.mul_(beta1).add_(grad_stiefel, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_stiefel, grad_stiefel, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                denom = exp_avg_sq.sqrt().add_(eps)
                update = exp_avg / denom

                # Update parameter while maintaining orthogonality
                Q = head_param - step_size * update
                U, _, V = torch.svd(Q)
                head_param.copy_(U @ V.t())

                # Update the original parameter tensor
                param[head_idx] = head_param
                p.data = param.view(original_shape)

        return loss 