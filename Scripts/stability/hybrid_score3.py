"""
Hybrid Score 3（Cho et al., Nat Commun 2025 类定义）:
  0.86 * z(ESM-2 序列打分) + 0.14 * z(半掩码 ESMFold 平均 pLDDT)

说明:
- 论文在 20% 数据上调权重；此处默认 0.86/0.14，并支持命令行修改。
- ESM-2: 采用「全序列可见、单遍前向」下各位置真实氨基酸 log p 的均值（与严格逐位 PLL 略有差别，
  但计算快、排序相关通常仍可用）。若需严格 PLL 可后续扩展。
- ESMFold: 对序列随机 50% 位置置为 X，重复 n_seeds 次，取 mean pLDDT 再对长度归一为「每残基均值」。

依赖示例:
  pip install fair-esm omegaconf "fair-esm[esmfold]"
  pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
  pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
  勿用 PyPI 的 pip install openfold（错误包）。
  OpenFold 从源码编译时若报 C++17：旧版 setup 用 -std=c++14，与 PyTorch 2.5+ 不兼容，需在克隆的 openfold 里把扩展的 c++14 改为 c++17 后再 pip install .（见 requirements_hybrid.txt）。
  若暂时装不上 OpenFold：可用 hybrid_score3_batch(..., skip_esmfold=True)，此时 hybrid 退化为 batch 内 z(ESM-2)，与论文含结构项的 Hybrid 不可直接对比。
"""
from __future__ import annotations

import random
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - m) / s


def esm2_sequence_logprob_mean(
    sequence: str,
    model,
    alphabet,
    device: torch.device,
) -> float:
    """单遍 MLM 头：对每个位置取真实氨基酸的 log prob，再对有效位置平均。"""
    batch_converter = alphabet.get_batch_converter()
    seq_clean = "".join(c for c in sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    if len(seq_clean) < 1:
        return float("nan")
    data = [("protein", seq_clean)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[], return_contacts=False)
        logits = out["logits"]  # [1, L+2, 33]
    # 对齐 alphabet：batch_tokens 含 BOS/EOS
    log_probs = torch.log_softmax(logits, dim=-1)
    # 位置 1..L 对应序列
    total = 0.0
    n = 0
    for i, aa in enumerate(seq_clean):
        tok_idx = batch_tokens[0, i + 1].item()
        # 标准氨基酸 token
        lp = log_probs[0, i + 1, tok_idx].item()
        total += lp
        n += 1
    return total / max(n, 1)


def esmfold_half_mask_mean_plddt(
    sequence: str,
    fold_model,
    device: torch.device,
    n_seeds: int = 8,
    mask_frac: float = 0.5,
    rng: Optional[random.Random] = None,
) -> float:
    """
    半掩码 ESMFold：随机将约 mask_frac 比例残基改为 'X'，重复 n_seeds，取平均 pLDDT。
    """
    if rng is None:
        rng = random.Random(0)
    seq_clean = "".join(c for c in sequence.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
    L = len(seq_clean)
    if L < 1:
        return float("nan")
    n_mask = max(1, int(round(L * mask_frac)))
    scores = []
    with torch.no_grad():
        for _ in range(n_seeds):
            idx = rng.sample(range(L), k=min(n_mask, L))
            chars = list(seq_clean)
            for j in idx:
                chars[j] = "X"
            masked = "".join(chars)
            try:
                out = fold_model.infer(masked)
            except Exception as e:
                warnings.warn(f"ESMFold infer failed: {e}")
                continue

            def _extract_plddt(o):
                if isinstance(o, dict):
                    if "mean_plddt" in o:
                        v = o["mean_plddt"]
                        return float(v.item() if torch.is_tensor(v) else v)
                    if "plddt" in o:
                        p = o["plddt"]
                        return float(p.mean().item() if torch.is_tensor(p) else np.mean(p))
                v = getattr(o, "mean_plddt", None) or getattr(o, "avg_plddt", None)
                if v is not None:
                    return float(v.item() if torch.is_tensor(v) else v)
                return None

            m = _extract_plddt(out)
            if m is None:
                warnings.warn("ESMFold output: could not read mean pLDDT; type=%s" % type(out))
                break
            scores.append(m)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _patch_deepspeed_utils_for_openfold() -> None:
    """DeepSpeed 0.14+ 不再提供 deepspeed.utils.is_initialized；旧版 OpenFold 仍引用。"""
    try:
        import deepspeed.utils as u

        if hasattr(u, "is_initialized"):
            return
        import deepspeed.comm as c

        fn = getattr(c, "is_initialized", None)
        if fn is not None:
            u.is_initialized = fn  # type: ignore[attr-defined]
        else:
            u.is_initialized = lambda: False  # type: ignore[attr-defined]
    except Exception:
        pass


def load_esm2(device: torch.device, model_name: str = "esm2_t33_650M_UR50D"):
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    return model, alphabet


def load_esmfold_v1(device: torch.device):
    _patch_deepspeed_utils_for_openfold()
    import esm

    model = esm.pretrained.esmfold_v1()
    model = model.eval().to(device)
    return model


def hybrid_score3_batch(
    sequences: List[str],
    w_esm: float = 0.86,
    w_plddt: float = 0.14,
    device: Optional[str] = None,
    esm2_name: str = "esm2_t33_650M_UR50D",
    n_esmfold_seeds: int = 8,
    esmfold_mask_frac: float = 0.5,
    seed: int = 0,
    skip_esmfold: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对多条序列计算 esm 均值 logprob、esmfold 半掩码 pLDDT、以及 hybrid（对两条子分数在 batch 内 z-score 后加权）。
    返回 (esm_scores, plddt_scores, hybrid, z_esm, z_plddt)。

    skip_esmfold=True：不加载 ESMFold/OpenFold，pLDDT 为 nan，z_plddt 为 0，hybrid 退化为 batch 内 z(ESM-2)（与论文 Hybrid 定义不同，仅作无结构项时的排序参考）。
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = random.Random(seed)
    nseq = len(sequences)

    print(f"[hybrid_score3] Loading ESM-2 ({esm2_name})...", flush=True)
    esm_model, alphabet = load_esm2(dev, esm2_name)
    fold_model = None
    if not skip_esmfold:
        print("[hybrid_score3] Loading ESMFold (大模型，可能 1～5 分钟无新输出)...", flush=True)
        fold_model = load_esmfold_v1(dev)

    print(f"[hybrid_score3] Scoring {nseq} sequences (每条含 ESM2 + 多次 ESMFold，全量可能极慢)...", flush=True)

    esm_list = []
    plddt_list = []
    for seq in tqdm(sequences, desc="hybrid_score3", unit="seq", mininterval=0.5):
        esm_list.append(esm2_sequence_logprob_mean(seq, esm_model, alphabet, dev))
        if skip_esmfold:
            plddt_list.append(float("nan"))
        else:
            plddt_list.append(
                esmfold_half_mask_mean_plddt(
                    seq, fold_model, dev, n_seeds=n_esmfold_seeds, mask_frac=esmfold_mask_frac, rng=rng
                )
            )

    esm_arr = np.array(esm_list, dtype=np.float64)
    plddt_arr = np.array(plddt_list, dtype=np.float64)
    z_e = _zscore(esm_arr)
    if skip_esmfold:
        warnings.warn(
            "skip_esmfold=True：未使用 ESMFold；hybrid_score3 列等于 batch 内 z(ESM-2)，"
            "与论文 0.86*z_esm+0.14*z_plddt 不一致。",
            stacklevel=2,
        )
        z_p = np.zeros_like(z_e)
        hybrid = z_e
    else:
        z_p = _zscore(plddt_arr)
        hybrid = w_esm * z_e + w_plddt * z_p
    return esm_arr, plddt_arr, hybrid, z_e, z_p
