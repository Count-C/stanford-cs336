import os
import torch
import tqdm
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, lr_cosine_schedule, gradient_clipping
from cs336_basics.data import get_batch, save_checkpoint, load_checkpoint
from cs336_basics.loss import cross_entropy


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
    num_batches: int = 100,
) -> float:
    """Run validation over num_batches and return average loss."""
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
        )
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def train():
    cfg = OmegaConf.load("config.yaml")
    torch.manual_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    print("PyTorch 编译时使用的 CUDA 版本:", torch.version.cuda)

    model = TransformerLM(
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        cfg.model.d_model,
        cfg.model.num_heads,
        cfg.model.d_ff,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    # Set up TensorBoard writer
    log_dir = Path(cfg.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir), flush_secs=120)

    # Load pre-tokenized training data and validation data
    train_data = np.memmap(cfg.data.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.data.val_path, dtype=np.uint16, mode="r")

    start = 0
    ckpt_file = getattr(cfg.checkpoint, "load_file", None)
    if ckpt_file is not None and os.path.exists(ckpt_file):
        start = load_checkpoint(ckpt_file, model, optimizer)

    model.train()
    progress_bar = tqdm.trange(start, cfg.train.max_iters, desc="Training")
    for step in progress_bar:
        # 1. Sample a batch of input sequences and next-token targets
        x, y = get_batch(
            train_data, cfg.train.batch_size, cfg.model.context_length, device
        )

        # 2. Forward pass — logits shape: (batch, context_length, vocab_size)
        logits = model(x)

        # 3. Compute cross-entropy loss (use custom implementation)
        loss = cross_entropy(
            logits.view(-1, cfg.model.vocab_size),
            y.view(-1),
        )

        # 4. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 5. Gradient clipping (if configured)
        if hasattr(cfg.train, "grad_clip") and cfg.train.grad_clip > 0:
            gradient_clipping(model.parameters(), cfg.train.grad_clip)

        # 6. Learning rate schedule: cosine with linear warmup
        lr = lr_cosine_schedule(
            step, cfg.train.lr, cfg.train.min_lr, cfg.train.warmup_iters, cfg.train.max_iters
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 7. Optimizer step
        optimizer.step()

        # 8. Logging — console + TensorBoard
        if step % cfg.logging.log_interval == 0 or step % cfg.logging.tb_interval == 0:
            loss_val = loss.item()
            perplexity = torch.exp(loss).item()
            # Compute total gradient norm for monitoring
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm ** 0.5

            # Console
            if step % cfg.logging.log_interval == 0:
                progress_bar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    ppl=f"{perplexity:.2f}",
                    lr=f"{lr:.2e}",
                    grad_norm=f"{total_norm:.2f}",
                )

            # TensorBoard
            if step % cfg.logging.tb_interval == 0:
                writer.add_scalar("loss/train", loss_val, step)
                writer.add_scalar("perplexity/train", perplexity, step)
                writer.add_scalar("lr", lr, step)
                writer.add_scalar("grad_norm", total_norm, step)

                val_loss = validate(
                    model,
                    val_data,
                    cfg.train.batch_size,
                    cfg.model.context_length,
                    cfg.model.vocab_size,
                    device,
                    num_batches=getattr(cfg.train, "eval_num_batches", 100)
                )
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                writer.add_scalar("loss/val", val_loss, step)
                writer.add_scalar("perplexity/val", val_ppl, step)
                progress_bar.write(
                    f"Step {step:>6d}  |  val_loss {val_loss:.4f}  |  val_ppl {val_ppl:.2f}"
                )

        # 9. Checkpointing
        if step > 0 and step % cfg.checkpoint.save_interval == 0:
            save_dir = Path(cfg.checkpoint.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model, optimizer, step, save_dir / f"checkpoint_{step}.pt"
            )

    writer.close()


if __name__ == "__main__":
    train()
