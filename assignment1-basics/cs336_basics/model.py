import math
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool

class Linear(nn.Module):
    def __init__(self, 
        d_in: int, 
        d_out: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = d_in
        self.out_features = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        # return einsum(x, self.weight, "b n d_in, d_out d_in -> b n d_out")
        return x @ self.weight.T
    

class Embedding(nn.Module):
    def __init__(self, 
        num_embeddings: int,
        embedding_dim: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ): 
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_emb"]:
        return self.weight[token_ids, :]


class RMSNorm(nn.Module):
    def __init__(self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms = torch.rsqrt(reduce(x.pow(2), "b n d_model -> b n 1", "mean") + self.eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = x * rms * self.weight
        return result.to(in_dtype) 

class SwiGLU(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))
        

def silu(x: Float[torch.Tensor, " ... d"]) -> Float[torch.Tensor, " ... d"]:
    # return x / (1 + torch.exp(-x))
    return x * torch.sigmoid(x)


class RoPE(nn.Module):
    def __init__(self, 
        max_seq_len: int,
        d_k: int,
        theta: float = 10000.0,
        device: torch.device | None = None
    ):
        super().__init__()
        # 它用于存储那些需要跟随模型一起保存、加载和迁移设备（如从 CPU 到 GPU），但不需要被优化器训练的数据
        # persistent=False 表示这个 buffer 不会被保存，也不会在加载模型时被加载
        self.register_buffer(
            "freq_cache", RoPE.__init_cache(max_seq_len, d_k, theta, device), persistent=False)
        self.freq_cache: Float[torch.Tensor, "2 max_seq_len half_d"]

    @staticmethod
    def __init_cache(
        max_seq_len: int,
        d_k: int,
        theta: float = 10000.0,
        device: torch.device | None = None
    ) -> Float[torch.Tensor, "2 max_seq_len half_d"]:
        # 计算每个位置旋转时需要的 theta_i,k = i / theta^(2k/d_k)
        # i 是位置索引：i = 0, 1, ..., max_seq_len-1
        # k 是维度索引: k = 0, 1, ..., d_k/2-1
        # theta 是一个超参数，通常设置为 10000。
        assert d_k % 2 == 0, "d_k must be even"
        d = torch.arange(0, d_k, 2, device=device) / d_k
        freqs = torch.tensor(theta, device=device) ** (-d)
        seq = torch.arange(max_seq_len, device=device)
        # freqs = seq[:, None] * freqs[None, :]
        freqs = einsum(seq, freqs, "s, f -> s f")
        sin, cos = torch.sin(freqs), torch.cos(freqs)
        return torch.stack((sin, cos), dim=0).to(device=device)

    def forward(self, 
        x: Float[torch.Tensor, "... d"], 
        token_positions: Int[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "... d"]:
        # 1. 按最后一维的奇偶索引分组（d_k需为偶数）
        x_1 = x[..., ::2]
        x_2 = x[..., 1::2]
        # 2. 获取当前序列长度对应的cos/sin值
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        sin, cos = self.freq_cache[:, token_positions, :]
        # 3. 应用旋转公式：将位置信息融入向量
        x1_rot = x_1 * cos - x_2 * sin
        x2_rot = x_1 * sin + x_2 * cos
        # 4. 重组维度：将奇偶分组合并回原d_k维度
        out = torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2) # (..., seq_len, d_k)
        return out
   

def softmax(x: Float[torch.Tensor, "... d"], dim: int = -1) -> Float[torch.Tensor, "... d"]:
    max_elems = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_elems)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: Float[torch.Tensor, "... n d_k"],
    key: Float[torch.Tensor, "... m d_k"],
    value: Float[torch.Tensor, "... m d_v"],
    mask: Bool[torch.Tensor, "..."] | None = None
) -> Float[torch.Tensor, "... n d_v"]:
    d_k = query.shape[-1]
    # scores = einsum(query, key, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    # output = einsum(attn_weights, value, "... n m, ... m d_v -> ... n d_v")
    output = attn_weights @ value
    return output
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_mask: bool = True,
        max_seq_len: int | None = None, theta: float = 10000.0
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_mask = use_mask
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.rope = RoPE(max_seq_len, self.d_k, theta) if max_seq_len is not None else None

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)

    def forward(self, 
        x: Float[torch.Tensor, "... d_model"],
        token_positions: Int[torch.Tensor, "..."] | None = None
    ) -> Float[torch.Tensor, "... d_model"]:
        batch, seq_len, d_model = x.shape
        assert d_model == self.d_model, "Input feature dimension must match d_model"
        
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        # query = rearrange(query, "b n (h d_k) -> b h n d_k", h=self.num_heads)
        # key = rearrange(key, "b n (h d_k) -> b h n d_k", h=self.num_heads)
        # value = rearrange(value, "b n (h d_v) -> b h n d_v", h=self.num_heads)
        query = query.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2) # b h n d_k
        key = key.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2) # b h n d_k
        value = value.view(batch, seq_len, self.num_heads, self.d_v).transpose(1, 2) # b h n d_v

        mask = None
        if self.use_mask:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=0).bool()

        if self.rope is not None:
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        attn_output = scaled_dot_product_attention(query, key, value, mask)
        # attn_output = rearrange(attn_output, "b h n d_v -> b n (h d_v)")
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.d_v)
        output = self.out_proj(attn_output)
        return output
        

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
        use_mask: bool = True, max_seq_len: int | None = None, theta: float = 10000.0
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_mask, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)
    
    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        x1 = self.attn(self.ln1(x))
        x2 = x + x1
        x3 = self.ffn(self.ln2(x2))
        return x2 + x3
    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
        d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=theta)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, token_ids: Int[torch.Tensor, "b n"]) -> Float[torch.Tensor, "b n vocab_size"]:
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
        