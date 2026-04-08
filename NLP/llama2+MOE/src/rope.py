import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    # 1. 计算频率：1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 2. 生成位置序列 t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 3. 计算相位：t 和 freqs 的外积
    freqs = torch.outer(t, freqs).float()
    # 4. 转换为复数形式 (cos(theta) + i*sin(theta))
    # freqs的形状为 (end, dim//2)，每个元素是一个实数频率值。我们使用 torch.polar 将这些实数频率转换为复数形式，其中实部为 cos(freqs)，虚部为 sin(freqs)。
    # polar函数的第一个参数是幅度，这里我们设置为1（即不改变频率的幅度），第二个参数是相位，即 freqs 中的值。结果 freqs_cis 的形状仍然是 (end, dim//2)，但每个元素现在是一个复数，表示旋转的相位。
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # x的形状为[批次大小, 序列长度, 注意力头数, 维度]
    # freqs_cis的形状为[序列长度, 维度//2]
    ndim = x.ndim   # ndim = 4
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 允许 GQA：Q/K 头数可不同，但最后一维 head_dim 应一致
    head_dim = xq.shape[-1]
    # 将 Q/K 向量视为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # 分别对 Q/K 广播以兼容不同头数
    freqs_q = reshape_for_broadcast(freqs_cis, xq_)
    freqs_k = reshape_for_broadcast(freqs_cis, xk_)
    # 复数乘法即为旋转
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, n_heads, n_kv_heads, head_dim = 4, 16, 8, 2, 16
    dim = n_heads * head_dim
    n_rep = n_heads // n_kv_heads

    # --- Test precompute_freqs_cis ---
    print("--- Test precompute_freqs_cis ---")
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len * 2)
    print("freqs_cis shape:", freqs_cis.shape)

    # --- Test apply_rotary_emb ---
    print("\n--- Test apply_rotary_emb ---")
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_slice = freqs_cis[:seq_len]
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_slice)
    print("xq shape (in/out):", xq.shape, xq_out.shape)
    print("xk shape (in/out):", xk.shape, xk_out.shape)


