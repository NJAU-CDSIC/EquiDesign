#!/usr/bin/env python3
"""Check pre-cutoff sequence homologs for PDB.chain targets.

The input FASTA is expected to use PDB.chain identifiers. Each query sequence is
compared with the corresponding official RCSB polymer sequence, and the output
reports sequence identity, coverage, release date, and the 40% identity flag.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests


def read_fasta(path: Path) -> Dict[str, str]:
    records: Dict[str, List[str]] = {}
    name = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0].split("|")[0]
            records[name] = []
        elif name is not None:
            records[name].append(line)
    return {k: "".join(v).upper() for k, v in records.items()}


def parse_target_id(target: str) -> Tuple[str, str]:
    m = re.match(r"^([0-9A-Za-z]{4})[._-]([A-Za-z0-9]+)", target)
    if not m:
        raise ValueError(f"Target id must look like PDB.chain, got {target!r}")
    return m.group(1).upper(), m.group(2)


def fetch_rcsb_entries(pdb_ids: Iterable[str]) -> Dict[str, dict]:
    ids = sorted(set(pdb_ids))
    query = """
    query($ids:[String!]!) {
      entries(entry_ids:$ids) {
        rcsb_id
        struct { title }
        rcsb_accession_info { initial_release_date }
        polymer_entities {
          rcsb_polymer_entity_container_identifiers { auth_asym_ids asym_ids }
          entity_poly { pdbx_seq_one_letter_code_can type }
        }
      }
    }
    """
    response = requests.post(
        "https://data.rcsb.org/graphql",
        json={"query": query, "variables": {"ids": ids}},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("errors"):
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return {entry["rcsb_id"].upper(): entry for entry in data["data"]["entries"]}


def local_align_identity(query: str, target: str) -> Tuple[float, float, float, int, bool]:
    """Return identity%, query coverage, target coverage, aln length, exact-substring flag."""
    if query in target:
        return 100.0, 1.0, len(query) / len(target), len(query), True

    # Smith-Waterman with a simple protein-agnostic scoring scheme.
    match, mismatch, gap = 2, -1, -2
    n, m = len(query), len(target)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[0] * (m + 1) for _ in range(n + 1)]
    best = (0, 0, 0)
    for i in range(1, n + 1):
        qi = query[i - 1]
        for j in range(1, m + 1):
            diag = H[i - 1][j - 1] + (match if qi == target[j - 1] else mismatch)
            up = H[i - 1][j] + gap
            left = H[i][j - 1] + gap
            val = max(0, diag, up, left)
            H[i][j] = val
            ptr[i][j] = 0 if val == 0 else (1 if val == diag else (2 if val == up else 3))
            if val > best[0]:
                best = (val, i, j)

    _, i, j = best
    matches = aln_len = q_used = t_used = 0
    while i > 0 and j > 0 and H[i][j] > 0:
        direction = ptr[i][j]
        if direction == 1:
            aln_len += 1
            q_used += 1
            t_used += 1
            matches += int(query[i - 1] == target[j - 1])
            i -= 1
            j -= 1
        elif direction == 2:
            aln_len += 1
            q_used += 1
            i -= 1
        elif direction == 3:
            aln_len += 1
            t_used += 1
            j -= 1
        else:
            break
    identity = 100.0 * matches / aln_len if aln_len else 0.0
    return identity, q_used / n if n else 0.0, t_used / m if m else 0.0, aln_len, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-fasta", required=True, type=Path)
    parser.add_argument("--cutoff", default="2021-09-30")
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    records = read_fasta(args.query_fasta)
    pdb_ids = [parse_target_id(name)[0] for name in records]
    entries = fetch_rcsb_entries(pdb_ids)

    rows = []
    for target_name, query_seq in records.items():
        pdb_id, chain_id = parse_target_id(target_name)
        entry = entries[pdb_id]
        release_date = entry["rcsb_accession_info"]["initial_release_date"][:10]
        best = None
        for entity in entry["polymer_entities"]:
            ids = entity["rcsb_polymer_entity_container_identifiers"]
            auth_chains = ids.get("auth_asym_ids") or []
            if chain_id not in auth_chains:
                continue
            target_seq = "".join(entity["entity_poly"]["pdbx_seq_one_letter_code_can"].split()).upper()
            identity, qcov, tcov, aln_len, exact = local_align_identity(query_seq, target_seq)
            best = {
                "protein_id": target_name,
                "matched_protein": f"{pdb_id.lower()}.{chain_id}",
                "release_date": release_date,
                "query_len": len(query_seq),
                "target_len": len(target_seq),
                "max_seq_identity_percent": f"{identity:.3f}",
                "query_coverage": f"{qcov:.3f}",
                "target_coverage": f"{tcov:.3f}",
                "alignment_len": aln_len,
                "exact_subsequence": exact,
                "released_before_cutoff": release_date <= args.cutoff,
                "ge_40_percent_identity": identity >= 40.0,
            }
            break
        if best is None:
            raise RuntimeError(f"No RCSB polymer entity found for {target_name}")
        rows.append(best)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
