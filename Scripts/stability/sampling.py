"""
自回归序列采样：与 model_utils.ProteinMPNN / ProteinEquiFormer 的 forward 解码顺序一致。
未解码位置使用氨基酸下标 20（GAP/'-'，与训练时 lookup 一致）。
"""
import torch
import torch.nn.functional as F


def _undouble_dataloader_batch_dim(x):
    if x is None:
        return None
    out = x
    while out.dim() >= 2 and out.shape[0] == 1 and out.shape[1] == 1:
        out = out.squeeze(0)
    return out


def autoregressive_sample(
    model,
    X,
    mask,
    chain_M,
    ss=None,
    temperature=0.0,
    seed=None,
):
    """
    X: [1, L, 4, 3], mask: [1, L], chain_M: [1, L], ss: [1, L, 3] or None
    temperature==0 -> 贪心；>0 -> 多项式采样。
    """
    device = X.device
    X = _undouble_dataloader_batch_dim(X)
    mask = _undouble_dataloader_batch_dim(mask)
    chain_M = _undouble_dataloader_batch_dim(chain_M)
    ss = _undouble_dataloader_batch_dim(ss)
    if X.dim() != 4:
        raise ValueError(f"X must be [B,L,4,3], got dim={X.dim()} shape={tuple(X.shape)}")
    B, L = X.shape[0], X.shape[1]
    if B != 1:
        raise ValueError("autoregressive_sample 当前仅支持 batch_size=1")

    if seed is not None:
        torch.manual_seed(seed)

    def _to_batch_seq_2d(name, t):
        if t.dim() == 1:
            if t.shape[0] != L:
                raise ValueError(f"{name} length {t.shape[0]} != X length {L}")
            return t.unsqueeze(0)
        if t.dim() == 2:
            if t.shape == (B, L):
                return t
            if B == 1 and t.shape == (1, L):
                return t
            if B == 1 and t.shape == (L, 1):
                return t.squeeze(-1).unsqueeze(0)
        raise ValueError(f"{name} shape {tuple(t.shape)} incompatible with X (B={B}, L={L})")

    mask = _to_batch_seq_2d("mask", mask)
    chain_M = _to_batch_seq_2d("chain_M", chain_M)

    if ss is not None and ss.dim() == 2 and ss.shape[-1] == 3:
        ss = ss.unsqueeze(0)

    S = torch.full((B, L), 20, dtype=torch.long, device=device)
    chain_M = chain_M * mask
    scoring = (chain_M + 0.0001) * torch.abs(torch.randn(chain_M.shape, device=device))
    decoding_order = torch.argsort(scoring, dim=-1)

    model.eval()
    with torch.no_grad():
        for step in range(L):
            if decoding_order.dim() == 1:
                pos = int(decoding_order[step].item())
            else:
                pos = int(decoding_order[0, step].item())
            if chain_M[0, pos] < 0.5:
                continue
            log_probs = model(X, S, mask, chain_M, ss)
            logits = log_probs[0, pos]
            if temperature is None or temperature <= 0:
                aa = int(torch.argmax(logits).item())
            else:
                prob = F.softmax(logits / temperature, dim=-1)
                aa = int(torch.multinomial(prob, num_samples=1).item())
            S[0, pos] = aa
    return S
