#!/usr/bin/env python3
"""Search or compute structural neighbors for PDB.chain targets.

Two modes are provided:
  1. self-hit mode: download the corresponding RCSB chain and report the
     self-hit TM-score. This is the appropriate complete-PDB, self-hit-allowed
     overlap check for pre-cutoff PDB chains.
  2. Foldseek PDB100 mode: submit the query chain to the Foldseek web API and
     report the best returned representative-library hit. This is useful for a
     non-self/representative structural-neighbor sanity check, but PDB100 is not
     identical to the complete PDB archive.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import tarfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


def parse_target_id(target: str) -> Tuple[str, str]:
    m = re.match(r"^([0-9A-Za-z]{4})[._-]([A-Za-z0-9]+)", target)
    if not m:
        raise ValueError(f"Target id must look like PDB.chain, got {target!r}")
    return m.group(1).upper(), m.group(2)


def read_targets(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def download_pdb_chain(pdb_id: str, chain_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pdb_id.lower()}.{chain_id}.pdb"
    if out.exists() and out.stat().st_size > 0:
        return out
    response = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
    response.raise_for_status()
    lines = []
    first_model = True
    seen_model = False
    for line in response.text.splitlines():
        if line.startswith("MODEL"):
            if seen_model:
                first_model = False
            seen_model = True
            continue
        if line.startswith("ENDMDL") and seen_model:
            break
        if not first_model:
            continue
        if line.startswith("ATOM  ") and len(line) > 21 and line[21].strip() == chain_id:
            lines.append(line)
    lines.append("END")
    out.write_text("\n".join(lines) + "\n")
    return out


def ca_coords_from_pdb(path: Path, chain_id: str | None = None) -> List[Tuple[float, float, float]]:
    coords = []
    seen = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if chain_id and line[21].strip() != chain_id:
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16].strip()
        if alt not in ("", "A"):
            continue
        key = (line[21], line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return coords


def kabsch_rmsd_and_tm(query: List[Tuple[float, float, float]], target: List[Tuple[float, float, float]], lnorm: int) -> Tuple[float, float]:
    # Identical self-hits can be reported exactly without an external aligner.
    if len(query) != len(target):
        raise ValueError("This lightweight self-hit calculator expects equal-length CA lists")
    if query == target:
        return 1.0, 0.0
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Non-identical structural alignment requires numpy") from exc
    p = np.array(query, dtype=float)
    q = np.array(target, dtype=float)
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    v, _, wt = np.linalg.svd(pc.T @ qc)
    det = np.sign(np.linalg.det(v @ wt))
    rot = v @ np.diag([1.0, 1.0, det]) @ wt
    diff = pc @ rot - qc
    dists = np.sqrt((diff * diff).sum(axis=1))
    d0 = 1.24 * ((lnorm - 15) ** (1 / 3)) - 1.8 if lnorm > 21 else 0.5
    tm = float(np.sum(1.0 / (1.0 + (dists / d0) ** 2)) / lnorm)
    rmsd = float(np.sqrt(np.mean(dists * dists)))
    return tm, rmsd


def fetch_release_dates(pdb_ids: Iterable[str]) -> Dict[str, str]:
    ids = sorted(set(x.upper() for x in pdb_ids))
    query = """
    query($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        rcsb_accession_info { initial_release_date }
      }
    }
    """
    response = requests.post(
        "https://data.rcsb.org/graphql",
        json={"query": query, "variables": {"ids": ids}},
        timeout=60,
    )
    response.raise_for_status()
    return {
        entry["rcsb_id"].upper(): entry["rcsb_accession_info"]["initial_release_date"][:10]
        for entry in response.json()["data"]["entries"]
    }


def run_self_hit(targets: List[str], cutoff: str, work_dir: Path) -> List[dict]:
    releases = fetch_release_dates(parse_target_id(t)[0] for t in targets)
    rows = []
    query_dir = work_dir / "query_pdbs_rcsb"
    for target in targets:
        pdb_id, chain_id = parse_target_id(target)
        pdb_file = download_pdb_chain(pdb_id, chain_id, query_dir)
        coords = ca_coords_from_pdb(pdb_file)
        tm, rmsd = kabsch_rmsd_and_tm(coords, coords, len(coords))
        release_date = releases[pdb_id]
        rows.append({
            "protein_id": target,
            "matched_neighbor": f"{pdb_id.lower()}.{chain_id}",
            "release_date": release_date,
            "ca_residues": len(coords),
            "max_tm_score": f"{tm:.6f}",
            "rmsd": f"{rmsd:.6f}",
            "released_before_cutoff": release_date <= cutoff,
            "tm_score_gt_0_5": tm > 0.5,
            "mode": "complete-PDB self-hit allowed, computed from RCSB chain coordinates",
        })
    return rows


def run_foldseek_pdb100(targets: List[str], cutoff: str, work_dir: Path) -> List[dict]:
    releases = fetch_release_dates(parse_target_id(t)[0] for t in targets)
    rows = []
    query_dir = work_dir / "query_pdbs_rcsb"
    result_dir = work_dir / "foldseek_api_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for target in targets:
        pdb_id, chain_id = parse_target_id(target)
        pdb_file = download_pdb_chain(pdb_id, chain_id, query_dir)
        with pdb_file.open("rb") as handle:
            response = requests.post(
                "https://search.foldseek.com/api/ticket",
                files={"q": (pdb_file.name, handle, "application/octet-stream")},
                data={"mode": "3diaa", "database[]": "pdb100"},
                timeout=90,
            )
        response.raise_for_status()
        ticket = response.json()
        ticket_id = ticket["id"]
        for _ in range(120):
            status = requests.get(f"https://search.foldseek.com/api/ticket/{ticket_id}", timeout=30).json()
            if status.get("status") == "COMPLETE":
                break
            if status.get("status") == "ERROR":
                raise RuntimeError(f"Foldseek job failed for {target}: {status}")
            time.sleep(5)
        download = requests.get(f"https://search.foldseek.com/api/result/download/{ticket_id}", timeout=120)
        download.raise_for_status()
        archive_path = result_dir / f"{target}.download.tar.gz"
        archive_path.write_bytes(download.content)
        with tarfile.open(fileobj=io.BytesIO(download.content), mode="r:gz") as archive:
            member = next(m.name for m in archive.getmembers() if m.name.endswith(".m8") and "report" not in m.name)
            first = archive.extractfile(member).readline().decode("utf-8", "ignore").rstrip("\n")
        parts = first.split("\t")
        hit = parts[1]
        m = re.match(r"^([0-9A-Za-z]{4})-.*?\.cif\.gz_([^\s]+)", hit)
        hit_pdb = m.group(1).lower() if m else ""
        hit_chain = m.group(2) if m else ""
        rows.append({
            "protein_id": target,
            "matched_neighbor": f"{hit_pdb}.{hit_chain}" if hit_pdb else hit,
            "foldseek_ticket_id": ticket_id,
            "seq_id_percent": parts[2],
            "prob": parts[10],
            "evalue": parts[11],
            "foldseek_score": parts[12],
            "query_release_date": releases[pdb_id],
            "query_released_before_cutoff": releases[pdb_id] <= cutoff,
            "mode": "Foldseek web API PDB100 representative search; not identical to complete PDB self-hit search",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, type=Path, help="Text file with one PDB.chain id per line")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--work-dir", default=Path("foldseek_neighbor_work"), type=Path)
    parser.add_argument("--cutoff", default="2021-09-30")
    parser.add_argument("--mode", choices=["self-hit", "foldseek-pdb100"], default="self-hit")
    args = parser.parse_args()

    targets = read_targets(args.targets)
    if args.mode == "self-hit":
        rows = run_self_hit(targets, args.cutoff, args.work_dir)
    else:
        rows = run_foldseek_pdb100(targets, args.cutoff, args.work_dir)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
