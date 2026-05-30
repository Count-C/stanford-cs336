import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf
from jaxtyping import Float

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM, softmax

def temperature_logits(
    logits: Float[torch.Tensor, "... vocab_size"],
    temperature: float
) -> Float[torch.Tensor, "... vocab_size"]:
    scaled_logits = logits / (temperature + 1e-8)
    return scaled_logits


def top_p_sampling(
    probs: Float[torch.Tensor, "... vocab_size"],
    top_p: float
) -> Float[torch.Tensor, "... vocab_size"]:
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1) # 计算前缀和
    mask = cumulative_probs > top_p
    # 保留第一个超过阈值的词
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    unsorted_probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return unsorted_probs


def get_device() -> torch.device:
    # 优先判断 CUDA (NVIDIA显卡)，其次判断 MPS (Mac GPU)，最后回退到 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    # Mac 上没有 CUDA 版本，所以这里加个简单的判断防止报错
    if torch.version.cuda:
        print("PyTorch 编译时使用的 CUDA 版本:", torch.version.cuda)

    print("当前使用的设备:", device)
    return device


@torch.no_grad()
def generate(prompt: str):
    cfg = OmegaConf.load("config.yaml")

    device = get_device()

    tokenizer = Tokenizer.from_files(cfg.generate.vocab_file, cfg.generate.merge_file, cfg.generate.special_tokens)

    model = TransformerLM(
        cfg.model.vocab_size,
        cfg.model.context_length,
        cfg.model.num_layers,
        cfg.model.d_model,
        cfg.model.num_heads,
        cfg.model.d_ff,
    ).to(device)

    checkpoint = torch.load(Path(cfg.generate.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    temperature = cfg.generate.temperature
    top_p = cfg.generate.top_k
    end_token = tokenizer.encode(cfg.generate.end_token)[0]

    model.eval()
    prompt_tokens = tokenizer.encode(prompt)
    context = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = []
    for _ in range(cfg.generate.max_length):
        logits = model(context)
        logits = temperature_logits(logits[:, -1, :], temperature)
        probs = softmax(logits)
        probs = top_p_sampling(probs, top_p)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token.item())
        if next_token.item() == end_token:
            break
        context = torch.cat([context, next_token], dim=1)
        if context.size(1) > cfg.model.context_length:
            context = context[:, -cfg.model.context_length:]

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("请在运行命令后直接加上你的 prompt！")
        print('例如：python generate.py "Once upon a time"')
        sys.exit(1)

    generated_text = generate(prompt)
    print(prompt + generated_text)