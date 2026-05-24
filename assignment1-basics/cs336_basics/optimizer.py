import torch
import math
from typing import Callable, Iterable, Optional
from jaxtyping import Float

class AdamW(torch.optim.Optimizer):
    def __init__(self, 
        params: Iterable, 
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01, 
        eps: float = 1e-8
    ):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data)) # 引用
                v = state.get("v", torch.zeros_like(p.data))

                grad = p.grad.data
                # p.data -= lr * weight_decay * p.data
                p.data.mul_(1 - lr * weight_decay)      # 原地操作速度更快
                lr *= (math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t)))
                # m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * torch.pow(grad, 2)
                # p.data -= lr * m / (torch.sqrt(v) + eps)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                p.data.addcdiv_(m, torch.sqrt(v) + eps, value=-lr)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
    

def lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, t_warm: int, t_cos: int):
    if t < t_warm:
        return alpha_max * t / t_warm
    elif t >= t_warm and t <= t_cos:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - t_warm) / (t_cos - t_warm)))
    else:
        return alpha_min
    

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, esp: float = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)
    for g in grads:
        norm.add_(g.pow(2).sum())
    norm.sqrt_()
    clip_coef = min(1, max_l2_norm / (norm + esp))
    for p in params:
        if p.grad is not None:
            p.grad.mul_(clip_coef)
