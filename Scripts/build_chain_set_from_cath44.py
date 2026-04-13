"""
Build chain_set.jsonl and chain_set_splits.json from CATH non-redundant S40 PDBs.

This is a copy of the original project script `training/build_chain_set_from_cath44.py`,
provided here for convenience when preparing datasets for EquiDesign.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

alpha_3 = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "GAP",
]
aa_3_1 = {b: a for a, b in zip("ARNDCQEGHILKMFPSTWYV-", alpha_3)}


def parse_pdb_backbone(pdb_path, atoms=("N", "CA", "C", "O"), chain_id=None):
    xyz_per_res = {}
    seq_per_res = {}
    first_chain = None

    with open(pdb_path, "rb") as f:
        for line in f:
            line = line.decode("utf-8", "ignore").rstrip()
            if line[:6] == "HETATM" and line[17:20] == "MSE":
                line = line.replace("HETATM", "ATOM  ").replace("MSE", "MET")
            if line[:4] != "ATOM":
                continue
            ch = line[21:22]
            if first_chain is None:
                first_chain = ch
            if chain_id is not None and ch != chain_id:
                continue
            if chain_id is None and ch != first_chain:
                continue
            atom = line[12:16].strip()
            resi = line[17:20]
            resn_str = line[22:27].strip()
            if resn_str[-1].isalpha():
                resn = int(resn_str[:-1]) - 1
            else:
                resn = int(resn_str) - 1
            x, y, z = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            if resn not in xyz_per_res:
                xyz_per_res[resn] = {}
                seq_per_res[resn] = resi
            if atom not in xyz_per_res[resn]:
                xyz_per_res[resn][atom] = [x, y, z]

    if not xyz_per_res:
        return None, None

    res_min, res_max = min(xyz_per_res), max(xyz_per_res)
    seq_list = []
    coords = {a: [] for a in atoms}
    for resn in range(res_min, res_max + 1):
        if resn not in seq_per_res:
            continue
        seq_list.append(aa_3_1.get(seq_per_res[resn], "-"))
        for a in atoms:
            if resn in xyz_per_res and a in xyz_per_res[resn]:
                coords[a].append(xyz_per_res[resn][a])
            else:
                coords[a].append([np.nan, np.nan, np.nan])
    seq_str = "".join(seq_list)
    coords_out = {a: coords[a] for a in atoms}
    return coords_out, seq_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list_file", type=str, required=True)
    ap.add_argument("--pdb_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./data_cath44")
    ap.add_argument("--split_ratio", type=float, nargs=3, default=[0.85, 0.05, 0.10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", type=str, default=".pdb")
    ap.add_argument("--list_column", type=int, default=0)
    args = ap.parse_args()

    with open(args.list_file, encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    domain_ids = []
    for line in raw_lines:
        parts = line.split()
        if len(parts) > args.list_column:
            did = parts[args.list_column]
            if (
                len(parts) >= 3
                and len(did) == 4
                and len(parts[args.list_column + 1]) == 1
                and parts[args.list_column + 2].isdigit()
            ):
                did = did + parts[args.list_column + 1] + parts[args.list_column + 2].zfill(2)
            domain_ids.append(did)
        else:
            domain_ids.append(line)

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out_dir, "chain_set.jsonl")
    split_path = os.path.join(args.out_dir, "chain_set_splits.json")

    # CATH 解压后 PDB 常在 dompdb/ 子目录下
    search_roots = [args.pdb_dir]
    dompdb = os.path.join(args.pdb_dir, "dompdb")
    if os.path.isdir(dompdb):
        search_roots.insert(0, dompdb)

    def find_pdb(did):
        bases = [did, did + args.suffix, did + ".ent", did.lower() + args.suffix]
        for root in search_roots:
            for base in bases:
                p = os.path.join(root, base)
                if os.path.isfile(p):
                    return p
                if len(did) >= 2:
                    p = os.path.join(root, did[:2], base)
                    if os.path.isfile(p):
                        return p
                if len(did) >= 4:
                    p = os.path.join(root, did[:4], base)
                    if os.path.isfile(p):
                        return p
                if len(did) >= 3:
                    p = os.path.join(root, did[1:3], base)
                    if os.path.isfile(p):
                        return p
        return None

    entries = []
    skipped = 0
    for did in domain_ids:
        pdb_path = find_pdb(did)
        if pdb_path is None:
            skipped += 1
            continue
        coords, seq = parse_pdb_backbone(pdb_path)
        if coords is None or not seq or len(seq) < 10:
            skipped += 1
            continue
        entries.append({"name": did, "seq": seq, "coords": coords})

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in entries:
            e_ = {
                "name": e["name"],
                "seq": e["seq"],
                "coords": {k: [list(x) for x in v] for k, v in e["coords"].items()},
            }
            f.write(json.dumps(e_) + "\n")

    names = [e["name"] for e in entries]
    random.seed(args.seed)
    random.shuffle(names)
    r1, r2 = args.split_ratio[0], args.split_ratio[0] + args.split_ratio[1]
    n = len(names)
    n1, n2 = int(n * r1), int(n * r2)
    splits = {"train": names[:n1], "validation": names[n1:n2], "test": names[n2:]}
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=0)

    print(f"Wrote {jsonl_path} ({len(entries)} entries), skipped {skipped}.")
    print(
        f"Wrote {split_path} (train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])})."
    )


if __name__ == "__main__":
    main()

