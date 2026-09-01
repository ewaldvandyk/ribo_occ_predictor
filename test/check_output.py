#!/usr/bin/env python3
"""Validate predict_ribo_occ_fasta.py output on the CDS test set.

Usage:  python check_output.py <output.bedgraph> [expected_cds.tsv]

Checks
------
1. Structure: every expected transcript present; intervals 0-based half-open,
   sorted, non-overlapping, contiguous from 0; total coverage == 3*K where
   K = min(codons in the three frames) for the known transcript length.
2. Function: within each annotated CDS, mean occupancy at in-frame positions
   (p % 3 == cds_start % 3) must exceed the mean at each of the two
   off-frame position sets — the model's frame preference on real CDS.

Exit 0 if all checks pass, 1 otherwise.
"""
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    bedgraph = Path(sys.argv[1])
    expected = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "expected_cds.tsv"

    # ── expected CDS table ────────────────────────────────────────────────
    exp: dict[str, dict] = {}
    with open(expected) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            exp[row["transcript_id"]] = {
                "length": int(row["tx_length_nt"]),
                "cds_start": int(row["cds_start"]),
                "cds_end": int(row["cds_end"]),
                "frame": int(row["cds_frame"]),
                "gene": row["gene_name"],
            }

    # ── parse bedGraph into per-transcript arrays ────────────────────────
    tracks: dict[str, np.ndarray] = {}
    intervals: dict[str, list[tuple[int, int]]] = {}
    for line in open(bedgraph):
        if line.startswith("#") or not line.strip():
            continue
        chrom, start, end, score = line.rstrip("\n").split("\t")
        start, end, score = int(start), int(end), float(score)
        if chrom not in tracks:
            n = exp[chrom]["length"] if chrom in exp else end
            tracks[chrom] = np.full(max(n, end), np.nan)
            intervals[chrom] = []
        if end > len(tracks[chrom]):
            tracks[chrom] = np.concatenate([tracks[chrom], np.full(end - len(tracks[chrom]), np.nan)])
        tracks[chrom][start:end] = score
        intervals[chrom].append((start, end))

    failures = []

    # ── 1. structural checks ─────────────────────────────────────────────
    for tx, e in exp.items():
        if tx not in tracks:
            failures.append(f"{tx}: missing from bedGraph output")
            continue
        iv = intervals[tx]
        n = e["length"]
        K = min(n // 3, (n - 1) // 3, (n - 2) // 3)
        covered = 3 * K
        ok_sorted = all(iv[i][1] <= iv[i + 1][0] for i in range(len(iv) - 1))
        ok_contig = iv[0][0] == 0 and all(iv[i][1] == iv[i + 1][0] for i in range(len(iv) - 1))
        last_end = iv[-1][1]
        if not ok_sorted:
            failures.append(f"{tx}: intervals not sorted/non-overlapping")
        if not ok_contig:
            failures.append(f"{tx}: intervals not contiguous from 0")
        if last_end != covered:
            failures.append(f"{tx}: coverage {last_end} != expected 3*K = {covered}")
        if np.isnan(tracks[tx][:covered]).any():
            failures.append(f"{tx}: gaps inside covered region")

    # ── 2. functional frame-preference check inside the CDS ──────────────
    print(f"{'transcript':24s} {'gene':10s} {'CDS':>13s} frame  in-frame  off+1  off+2  verdict")
    for tx, e in exp.items():
        if tx not in tracks:
            continue
        occ = tracks[tx]
        s, t, fr = e["cds_start"], e["cds_end"], e["frame"]
        pos = np.arange(s, min(t, len(occ)))
        means = {}
        for off in range(3):
            sel = occ[pos[(pos % 3) == ((fr + off) % 3)]]
            sel = sel[~np.isnan(sel)]
            means[off] = float(sel.mean()) if len(sel) else float("nan")
        ok = means[0] > means[1] and means[0] > means[2]
        if not ok:
            failures.append(
                f"{tx} ({e['gene']}): in-frame CDS occupancy {means[0]:.4f} does not "
                f"exceed off-frame ({means[1]:.4f}, {means[2]:.4f})"
            )
        print(f"{tx:24s} {e['gene']:10s} {s:6d}-{t:<6d} {fr}     "
              f"{means[0]:8.4f} {means[1]:6.4f} {means[2]:6.4f}  {'PASS' if ok else 'FAIL'}")

    if failures:
        print("\nFAILURES:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
