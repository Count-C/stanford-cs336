import torch
from jaxtyping import Float, Int
from cs336_basics.model import softmax

def cross_entropy(
    inputs: Float[torch.Tensor, "... d_vocab"], 
    targets: Int[torch.Tensor, "..."]
) -> Float[torch.Tensor, "..."]:
    # log(softmax(x)) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    max_elems = torch.max(inputs, dim=-1, keepdim=True).values
    exp_sum = torch.sum(torch.exp(inputs - max_elems), dim=-1, keepdim=True)
    neg_log_prob = -(inputs - max_elems - torch.log(exp_sum))
    return neg_log_prob.gather(-1, targets.unsqueeze(-1)).mean()