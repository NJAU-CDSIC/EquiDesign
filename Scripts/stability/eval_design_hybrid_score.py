#!/usr/bin/env python3
"""Design sequences and compute Hybrid Score 3."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# training/ 为工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sampling import autoregressive_sample


def write_fasta_from_design_csv(
    csv_path: str,
    fasta_path: str,
    name_col: str = "name",
    seq_col: str = "seq_designed",
) -> None:
    """从 design 产出的 CSV 写出 FASTA（一条 name 一行序列）。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    out_dir = os.path.dirname(os.path.abspath(fasta_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as f:
        for row in rows:
            name = row.get(name_col, "")
            seq = row.get(seq_col, "")
            f.write(f">{name}\n{seq}\n")
    print(f"Wrote sequences FASTA: {fasta_path}")


def write_hybrid_scores_only(
    full_csv_path: str,
    scores_path: str,
) -> None:
    """从带 hybrid 列的完整 CSV 写出仅分数列的表（便于单独分析）。"""
    score_fields = [
        "name",
        "length",
        "esm2_mean_logprob",
        "esmfold_halfmask_mean_plddt",
        "hybrid_score3",
        "z_esm2",
        "z_plddt",
    ]
    rows_out = []
    with open(full_csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out = {}
            for k in score_fields:
                if k == "length" and k not in row:
                    continue
                out[k] = row.get(k, "")
            rows_out.append(out)

    out_dir = os.path.dirname(os.path.abspath(scores_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    # 若 CSV 无 length 列则只写有的列
    use_fields = [k for k in score_fields if rows_out and k in rows_out[0]]
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=use_fields, extrasaction="ignore")
        w.writeheader()
        for row in rows_out:
            w.writerow({k: row.get(k, "") for k in use_fields})
    print(f"Wrote hybrid scores: {scores_path}")


def index_to_sequence(S: torch.Tensor) -> str:
    idx_to_aa = "ACDEFGHIKLMNPQRSTVWY"
    S = S.view(-1).cpu().numpy()
    out = []
    for i in S:
        i = int(i)
        if 0 <= i < 20:
            out.append(idx_to_aa[i])
        else:
            out.append("X")
    return "".join(out)


def _entry_to_tensors(entry, device):
    """单条 jsonl entry -> X, mask, ss 与 DynamicLoader 一致 [1,L,...]"""
    seq = entry["seq"]
    l = len(seq)
    x = entry["coords"]
    if isinstance(x, dict):
        x = np.stack([x[c] for c in ["N", "CA", "C", "O"]], axis=1)
    X = np.zeros((1, l, 4, 3), dtype=np.float32)
    X[0, :l] = x
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.float32)
    X[isnan] = 0.0
    X = np.nan_to_num(X)

    ss_str = entry.get("ss", "C" * l)
    ss_map = {"H": 0, "E": 1, "C": 2}
    SS = np.zeros((1, l, 3), dtype=np.float32)
    for j, c in enumerate(ss_str[:l]):
        k = ss_map.get(c, 2)
        SS[0, j, k] = 1.0

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)
    ss_t = torch.tensor(SS, dtype=torch.float32, device=device)
    return X_t, mask_t, ss_t


def load_test_entries(jsonl_file: str, split_file: str, ss_file: str | None):
    with open(split_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    test_names = set(splits["test"])

    ss_dict = {}
    if ss_file and os.path.isfile(ss_file):
        with open(ss_file, "r", encoding="utf-8") as f:
            ss_dict = json.load(f)
        print(f"Loaded SS for {len(ss_dict)} chains from {ss_file}")
    elif ss_file:
        print(f"Note: ss_file not found: {ss_file}, using coil")

    entries = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["name"] in test_names:
                for k, v in e["coords"].items():
                    e["coords"][k] = np.asarray(v)
                e["ss"] = ss_dict.get(e["name"], "C" * len(e["seq"]))
                entries.append(e)
    return entries


def build_model_equidesign(args, device):
    from model_utils import EquiDesign
    return EquiDesign(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_encoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=0.0,
        equiformer_out_vector=getattr(args, "equiformer_out_vector", 0),
    ).to(device)


def cmd_design(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ss_path = args.ss_file
    if not ss_path:
        ss_path = os.path.normpath(
            os.path.join(os.path.dirname(args.jsonl_file), os.path.basename(args.ss_file_default))
        )

    entries = load_test_entries(args.jsonl_file, args.split_file, ss_path if os.path.isfile(ss_path) else None)
    if args.max_samples:
        entries = entries[: args.max_samples]
    print(f"Test structures to design: {len(entries)}")

    model = build_model_equidesign(args, device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["name", "seq_native", "seq_designed", "length", "sample_seed"]
        )
        for e in tqdm(entries):
            X, mask, ss = _entry_to_tensors(e, device)
            chain_M = mask.clone()
            seed = args.sample_seed_base + hash(e["name"]) % 10000
            S = autoregressive_sample(
                model, X, mask, chain_M, ss=ss, temperature=args.temperature, seed=seed
            )
            L = int(mask.sum().item())
            des = index_to_sequence(S[0, :L])
            native = e["seq"]
            w.writerow([e["name"], native, des, L, seed])
            f.flush()
    print(f"Wrote {args.out_csv}")
    of = (getattr(args, "out_fasta", "") or "").strip()
    if of:
        write_fasta_from_design_csv(args.out_csv, of)


def _fmt_float_cell(x) -> str:
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return "nan"
        return f"{xf:.6f}"
    except (TypeError, ValueError):
        return str(x)


def cmd_hybrid(args):
    import hybrid_score3 as h3

    rows = []
    with open(args.in_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if args.max_samples:
        rows = rows[: args.max_samples]

    if not rows:
        print("No rows in input CSV; nothing to score.")
        return

    seqs = [r[args.seq_column] for r in rows]
    device = args.hybrid_device or ("cuda" if torch.cuda.is_available() else "cpu")
    esm_arr, plddt_arr, hybrid, z_e, z_p = h3.hybrid_score3_batch(
        seqs,
        w_esm=args.w_esm,
        w_plddt=args.w_plddt,
        device=device,
        esm2_name=args.esm2_model,
        n_esmfold_seeds=args.esmfold_seeds,
        esmfold_mask_frac=args.esmfold_mask_frac,
        seed=args.hybrid_seed,
        skip_esmfold=getattr(args, "skip_esmfold", False),
    )

    fieldnames = list(rows[0].keys()) + [
        "esm2_mean_logprob",
        "esmfold_halfmask_mean_plddt",
        "hybrid_score3",
        "z_esm2",
        "z_plddt",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            row = dict(row)
            row["esm2_mean_logprob"] = _fmt_float_cell(esm_arr[i])
            row["esmfold_halfmask_mean_plddt"] = _fmt_float_cell(plddt_arr[i])
            row["hybrid_score3"] = _fmt_float_cell(hybrid[i])
            row["z_esm2"] = _fmt_float_cell(z_e[i])
            row["z_plddt"] = _fmt_float_cell(z_p[i])
            w.writerow(row)
    print(f"Wrote {args.out_csv}")
    os_path = (getattr(args, "out_scores", "") or "").strip()
    if os_path:
        write_hybrid_scores_only(args.out_csv, os_path)


def cmd_all(args):
    stem = os.path.splitext(os.path.abspath(args.out_csv))[0]
    of = (getattr(args, "out_fasta", "") or "").strip()
    if not of:
        args.out_fasta = stem + "_sequences.fasta"
    else:
        args.out_fasta = of
    os_path = (getattr(args, "out_scores", "") or "").strip()
    if not os_path:
        args.out_scores = stem + "_hybrid_scores.csv"
    else:
        args.out_scores = os_path

    base, ext = os.path.splitext(args.out_csv)
    mid = base + "_designed_only" + ext
    args_mid = argparse.Namespace(**vars(args))
    args_mid.out_csv = mid
    cmd_design(args_mid)

    h_args = argparse.Namespace(
        in_csv=mid,
        out_csv=args.out_csv,
        seq_column="seq_designed",
        max_samples=args.max_samples,
        w_esm=args.w_esm,
        w_plddt=args.w_plddt,
        hybrid_device=args.hybrid_device,
        esm2_model=args.esm2_model,
        esmfold_seeds=args.esmfold_seeds,
        esmfold_mask_frac=args.esmfold_mask_frac,
        hybrid_seed=args.hybrid_seed,
        out_scores=args.out_scores,
        skip_esmfold=getattr(args, "skip_esmfold", False),
    )
    cmd_hybrid(h_args)


def main():
    p = argparse.ArgumentParser(description="Test-set design + Hybrid Score 3")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_model_args(sp):
        sp.add_argument("--hidden_dim", type=int, default=128)
        sp.add_argument("--num_encoder_layers", type=int, default=3)
        sp.add_argument("--num_decoder_layers", type=int, default=3)
        sp.add_argument("--num_neighbors", type=int, default=48)
        sp.add_argument("--dropout", type=float, default=0.3)
        sp.add_argument("--equiformer_out_vector", type=int, default=0)

    # design
    d = sub.add_parser("design", help="仅生成测试集序列")
    d.add_argument("--jsonl_file", type=str, required=True)
    d.add_argument("--split_file", type=str, required=True)
    d.add_argument("--checkpoint", type=str, default="../../EquiDesign_code/Model/model_weights/best_model.pt")
    d.add_argument("--out_csv", type=str, required=True)
    d.add_argument("--model_type", type=str, choices=["equidesign"], default="equidesign")
    d.add_argument("--ss_file", type=str, default="", help="默认自动用 jsonl 同目录 chain_set_ss.json")
    d.add_argument("--ss_file_default", type=str, default="chain_set_ss.json")
    d.add_argument("--max_samples", type=int, default=0, help=">0 时只跑前 N 条（调试）")
    d.add_argument("--temperature", type=float, default=0.0)
    d.add_argument("--sample_seed_base", type=int, default=0)
    d.add_argument(
        "--out_fasta",
        type=str,
        default="",
        help="可选：另存 FASTA（>name + 生成序列），与 out_csv 内容一致",
    )
    add_model_args(d)

    # hybrid
    h = sub.add_parser("hybrid", help="对 CSV 中序列列算 Hybrid 3")
    h.add_argument("--in_csv", type=str, required=True)
    h.add_argument("--out_csv", type=str, required=True)
    h.add_argument("--seq_column", type=str, default="seq_designed")
    h.add_argument("--max_samples", type=int, default=0)
    h.add_argument("--w_esm", type=float, default=0.86)
    h.add_argument("--w_plddt", type=float, default=0.14)
    h.add_argument("--hybrid_device", type=str, default=None)
    h.add_argument("--esm2_model", type=str, default="esm2_t33_650M_UR50D")
    h.add_argument("--esmfold_seeds", type=int, default=8)
    h.add_argument("--esmfold_mask_frac", type=float, default=0.5)
    h.add_argument("--hybrid_seed", type=int, default=0)
    h.add_argument(
        "--out_scores",
        type=str,
        default="",
        help="可选：另存仅含 hybrid 各分数列的 CSV",
    )
    h.add_argument(
        "--skip_esmfold",
        action="store_true",
        help="不加载 ESMFold（无需 OpenFold）；hybrid 列退化为 batch 内 z(ESM-2)，与论文含 pLDDT 的 Hybrid 不同",
    )

    # all
    a = sub.add_parser("all", help="design 再 hybrid（中间 csv 同 out 前缀）")
    a.add_argument("--jsonl_file", type=str, required=True)
    a.add_argument("--split_file", type=str, required=True)
    a.add_argument("--checkpoint", type=str, default="../../EquiDesign_code/Model/model_weights/best_model.pt")
    a.add_argument("--out_csv", type=str, required=True)
    a.add_argument("--model_type", type=str, choices=["equidesign"], default="equidesign")
    a.add_argument("--ss_file", type=str, default="")
    a.add_argument("--ss_file_default", type=str, default="chain_set_ss.json")
    a.add_argument("--max_samples", type=int, default=0)
    a.add_argument("--temperature", type=float, default=0.0)
    a.add_argument("--sample_seed_base", type=int, default=0)
    a.add_argument("--w_esm", type=float, default=0.86)
    a.add_argument("--w_plddt", type=float, default=0.14)
    a.add_argument("--hybrid_device", type=str, default=None)
    a.add_argument("--esm2_model", type=str, default="esm2_t33_650M_UR50D")
    a.add_argument("--esmfold_seeds", type=int, default=8)
    a.add_argument("--esmfold_mask_frac", type=float, default=0.5)
    a.add_argument("--hybrid_seed", type=int, default=0)
    a.add_argument(
        "--out_fasta",
        type=str,
        default="",
        help="序列 FASTA；留空则自动为 <out_csv 同名>_sequences.fasta",
    )
    a.add_argument(
        "--out_scores",
        type=str,
        default="",
        help="仅分数 CSV；留空则自动为 <out_csv 同名>_hybrid_scores.csv",
    )
    a.add_argument(
        "--skip_esmfold",
        action="store_true",
        help="同 hybrid 子命令的 --skip_esmfold",
    )
    add_model_args(a)

    args = p.parse_args()
    if args.cmd == "design":
        if args.max_samples == 0:
            args.max_samples = None
        cmd_design(args)
    elif args.cmd == "hybrid":
        if args.max_samples == 0:
            args.max_samples = None
        cmd_hybrid(args)
    else:
        if args.max_samples == 0:
            args.max_samples = None
        cmd_all(args)


if __name__ == "__main__":
    main()
