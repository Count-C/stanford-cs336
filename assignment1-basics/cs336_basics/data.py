import os
import typing
import torch
import numpy as np
from numpy.typing import NDArray

def get_batch(dataset: NDArray, batch_size: int, context_length: int, device: torch.device):
    '''
            Write a function that takes a numpy array 𝑥 (integer array with token IDs), a
        batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns
        a pair of tensors: the sampled input sequences and the corresponding next-token targets. 
    '''
    start_idxs = torch.randint(len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.tensor(dataset[idx: idx + context_length], dtype=int) for idx in start_idxs]).to(device)
    y = torch.stack([torch.tensor(dataset[idx + 1: idx + context_length + 1], dtype=int) for idx in start_idxs]).to(device)
    return (x, y)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    ):
    checkpoint = {
        "epoch": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
    ):
    '''
        should load a checkpoint from src (path or file-like
        object), and then recover the model and optimizer states from that checkpoint. Your function
        should return the iteration number that was saved to the checkpoint. You can use
        torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
        load_state_dict method in both the model and optimizer to return them to their previous
        states.
    '''
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
