#!/usr/bin/env python3
"""predict_ribo_occ_fasta.py — FASTA → per-nucleotide ribosome-occupancy bedGraph.

Standalone command-line tool (no aifofomo checkout required; only torch + numpy)
that runs a trained RibosomeTranslator checkpoint on every transcript sequence
in a FASTA file and writes nucleotide-resolution occupancy in bedGraph format.

The model predicts per-codon occupancy in all three reading frames; the three
frame tracks are interleaved into a single per-nucleotide track:

    position 3k   <- frame_0[k]   (codon starting at nt 3k)
    position 3k+1 <- frame_1[k]   (codon starting at nt 3k+1)
    position 3k+2 <- frame_2[k]   (codon starting at nt 3k+2)

Output columns (bedGraph, transcript-relative, 0-based half-open):

    <transcript_id> <start> <end> <occupancy>

Consecutive positions with an identical score are merged into one interval
(standard bedGraph practice). Coordinates are along the transcript itself:
the "chrom" column is the FASTA record ID (first whitespace-delimited token
of the header line).

Usage
-----
    python predict_ribo_occ_fasta.py transcripts.fasta \
        --model /path/to/model_500000.pt \
        -o occupancy.bedgraph

    # or stream to stdout for pipeline use (logs go to stderr):
    python predict_ribo_occ_fasta.py transcripts.fasta -m model.pt > occ.bedgraph

Notes
-----
* Both DNA (T) and RNA (U) sequences are accepted; T is converted to U.
* Codons containing characters other than A/C/G/U (e.g. N) are mapped to the
  PAD token; the model still emits a prediction at those positions.
* Transcripts longer than the model's positional-embedding window
  (max_seq_len codons, i.e. 3*max_seq_len nt — derived from the checkpoint at
  load time, e.g. 12,000 nt for a 4000-codon model) are SKIPPED by default and
  recorded as '# SKIPPED' marker lines in the output, so downstream consumers
  can see which transcripts were ignored. Pass --long-transcripts truncate to
  instead predict the 5' window (recorded as a '# TRUNCATED' marker line).
* The last 2-4 nt of each transcript get no value (no complete codon starts
  there in every frame); bedGraph simply has no rows for those positions.

Dependencies: Python >= 3.10, torch, numpy.
"""

from __future__ import annotations

__version__ = "1.0.0"

import argparse
import gzip
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, TextIO, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# 1. Codon vocabulary and tokenisation
#    (mirrors aifofomo/common/biology/codonizer.py)
# ===========================================================================

_NUCLEOTIDES = ("A", "C", "G", "U")
_ALL_CODONS: list[str] = ["".join(c) for c in itertools.product(_NUCLEOTIDES, repeat=3)]
_CODON_TO_IDX: dict[str, int] = {codon: idx for idx, codon in enumerate(_ALL_CODONS)}

PAD_IDX: int = 64          # 64 possible codons → PAD = index 64
VOCAB_SIZE: int = 65       # 64 codons + 1 PAD


def _codonize(sequence: str, frame_offset: int = 0) -> list[int]:
    """Convert a nucleotide string to codon token IDs for one reading frame."""
    sub = sequence[frame_offset:]
    return [
        _CODON_TO_IDX.get(sub[i : i + 3], PAD_IDX)
        for i in range(0, len(sub) - 2, 3)
    ]


def _codonize_three_frames(sequence: str) -> tuple[list[int], list[int], list[int]]:
    """Return codon token lists for all three reading frames."""
    return (_codonize(sequence, 0), _codonize(sequence, 1), _codonize(sequence, 2))


# ===========================================================================
# 2. Neural-network building blocks
#    (mirrors aifofomo/models/layers/ and aifofomo/models/modules/attention.py)
# ===========================================================================

class DropPath(nn.Module):
    """Stochastic depth — identity during inference."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5,
                 inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """Two-layer feed-forward block."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 proj_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VisionTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, proj_bias: bool = True, ffn_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 init_values=None, drop_path: float = 0.0,
                 act_layer: Callable[..., nn.Module] = nn.GELU,
                 norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              proj_bias=proj_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ===========================================================================
# 3. RibosomeTranslator model (encoder + frame-occupancy head only)
# ===========================================================================

class RibosomeTranslator(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, mlp_ratio: float = 4.0,
                 max_seq_len: int = 2048,
                 init_values: Optional[float] = None,
                 use_frame_embed: bool = False) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len

        self.codon_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.frame_embed = nn.Embedding(3, embed_dim) if use_frame_embed else None

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, init_values=init_values,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.frame_head = nn.Linear(embed_dim, 1)

    def _encode(self, codon_tokens: torch.Tensor, codon_mask: torch.Tensor,
                frame_ids: torch.Tensor) -> torch.Tensor:
        _, L = codon_tokens.shape
        x = self.codon_embed(codon_tokens)
        x = x + self.pos_embed[:, :L, :]
        if self.frame_embed is not None:
            x = x + self.frame_embed(frame_ids).unsqueeze(1)
        x = x * codon_mask.unsqueeze(-1).float()
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ===========================================================================
# 4. Model loading
# ===========================================================================

def _infer_model_config(state_dict: dict, num_heads: Optional[int] = None) -> dict:
    """Infer model hyperparameters from state_dict tensor shapes.

    num_heads is not recoverable from shapes (qkv rows are always 3*embed_dim);
    we assume head_dim = 64 unless --num-heads is given explicitly.
    """
    embed_dim = state_dict["codon_embed.weight"].shape[1]
    max_seq_len = state_dict["pos_embed"].shape[1]
    vocab_size = state_dict["codon_embed.weight"].shape[0]
    num_layers = sum(1 for k in state_dict
                     if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    inferred = num_heads is None
    if inferred:
        qkv_rows = state_dict["blocks.0.attn.qkv.weight"].shape[0]
        num_heads = qkv_rows // (3 * 64)   # assumes head_dim = 64
    if num_heads < 1 or embed_dim % num_heads != 0:
        if inferred:
            raise ValueError(
                f"Cannot infer num_heads (embed_dim={embed_dim}); "
                f"pass it explicitly with --num-heads."
            )
        raise ValueError(
            f"--num-heads {num_heads} is invalid: embed_dim={embed_dim} "
            f"must be divisible by it."
        )
    use_frame_embed = any(k.startswith("frame_embed") for k in state_dict)
    return dict(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                num_layers=num_layers, max_seq_len=max_seq_len,
                use_frame_embed=use_frame_embed, num_heads_inferred=inferred)


def load_model(checkpoint_path: str | Path, device: torch.device,
               num_heads: Optional[int] = None) -> tuple[RibosomeTranslator, int]:
    """Load a RibosomeTranslator checkpoint; returns (model, max_seq_len)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    cfg = _infer_model_config(state_dict, num_heads=num_heads)

    model = RibosomeTranslator(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"], use_frame_embed=cfg["use_frame_embed"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    heads_note = (" (inferred assuming head_dim=64; pass --num-heads if your "
                  "checkpoint was trained with a different head_dim)"
                  if cfg["num_heads_inferred"] else "")
    print(f"[predict_ribo_occ_fasta] loaded checkpoint {checkpoint_path} "
          f"(iteration {ckpt.get('iteration', '?') if isinstance(ckpt, dict) else '?'}, "
          f"embed_dim={cfg['embed_dim']}, num_layers={cfg['num_layers']}, "
          f"num_heads={cfg['num_heads']}{heads_note}, "
          f"max_seq_len={cfg['max_seq_len']} codons)",
          file=sys.stderr)
    return model, cfg["max_seq_len"]


# ===========================================================================
# 5. FASTA parsing
# ===========================================================================

def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield (record_id, sequence) tuples from a FASTA file (.gz supported).

    The record ID is the first whitespace-delimited token of the header line.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    chunks: list[str] = []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0] if len(line) > 1 else ""
                if not header:
                    raise ValueError(f"FASTA record with empty header in {path}")
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"FASTA parse error: sequence before first '>' header in {path}")
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


# ===========================================================================
# 6. Prediction
# ===========================================================================

def predict_interleaved(sequence: str, model: RibosomeTranslator,
                        max_seq_len: int, device: torch.device) -> np.ndarray:
    """Predict per-nucleotide interleaved occupancy for one RNA sequence.

    Returns an array of length 3*K where K = min(codons in frames 0/1/2,
    capped at max_seq_len). Position 3k+f holds the occupancy of the codon
    starting at nucleotide 3k+f (reading frame f).
    """
    frames = _codonize_three_frames(sequence)
    n_valid = [min(len(f), max_seq_len) for f in frames]
    K = min(n_valid)
    if K == 0:
        return np.empty(0, dtype=np.float32)
    L = max(n_valid)

    tokens = torch.full((3, L), PAD_IDX, dtype=torch.long)
    mask = torch.zeros(3, L, dtype=torch.bool)
    for fi, frame_tokens in enumerate(frames):
        n = n_valid[fi]
        tokens[fi, :n] = torch.tensor(frame_tokens[:n], dtype=torch.long)
        mask[fi, :n] = True

    tokens = tokens.to(device)
    mask = mask.to(device)
    frame_ids = torch.arange(3, device=device)

    with torch.inference_mode():
        hidden = model._encode(tokens, mask, frame_ids)      # [3, L, D]
        logits = model.frame_head(hidden).squeeze(-1)        # [3, L]
        occupancy = F.softplus(logits)                       # [3, L]

    occ = occupancy.float().cpu().numpy()                    # [3, L]

    interleaved = np.empty(3 * K, dtype=np.float32)
    interleaved[0::3] = occ[0, :K]
    interleaved[1::3] = occ[1, :K]
    interleaved[2::3] = occ[2, :K]
    return interleaved


# ===========================================================================
# 7. bedGraph output
# ===========================================================================

def write_bedgraph(out: TextIO, transcript_id: str, values: np.ndarray,
                   precision: int = 6) -> int:
    """Write per-nucleotide values as bedGraph intervals; returns rows written.

    Consecutive positions with an identical (formatted) score are merged into
    one interval. Coordinates are 0-based half-open along the transcript.
    """
    n_rows = 0
    run_start = 0
    run_score: Optional[str] = None
    for i, v in enumerate(values):
        score = f"{v:.{precision}g}"
        if score != run_score:
            if run_score is not None:
                out.write(f"{transcript_id}\t{run_start}\t{i}\t{run_score}\n")
                n_rows += 1
            run_start = i
            run_score = score
    if run_score is not None:
        out.write(f"{transcript_id}\t{run_start}\t{len(values)}\t{run_score}\n")
        n_rows += 1
    return n_rows


# ===========================================================================
# 8. Main
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="predict_ribo_occ_fasta.py",
        description="Predict per-nucleotide ribosome occupancy for every "
                    "transcript in a FASTA file using a trained "
                    "RibosomeTranslator model; output in bedGraph format "
                    "(transcript-relative coordinates).",
    )
    parser.add_argument("fasta", help="Input FASTA file with transcript "
                        "nucleotide sequences (DNA or RNA; .gz supported)")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to trained RibosomeTranslator .pt checkpoint")
    parser.add_argument("-o", "--output", default="-",
                        help="Output bedGraph file (default: '-' = stdout)")
    parser.add_argument("--device", default=None,
                        help="Torch device (cpu, cuda, mps); default: cuda if "
                             "available, else cpu")
    parser.add_argument("--long-transcripts", choices=("skip", "truncate"),
                        default="skip",
                        help="Transcripts longer than the model window "
                             "(3*max_seq_len nt, derived from the checkpoint): "
                             "'skip' (default) ignores them and writes a "
                             "'# SKIPPED' marker line to the output; "
                             "'truncate' predicts the 5' window and writes a "
                             "'# TRUNCATED' marker line.")
    parser.add_argument("--num-heads", type=int, default=None,
                        help="Number of attention heads (default: inferred "
                             "assuming head_dim=64)")
    parser.add_argument("--precision", type=int, default=6,
                        help="Significant digits for scores (default: 6)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-transcript progress on stderr "
                             "(warnings and errors are always shown)")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        print(f"ERROR: FASTA file not found: {fasta_path}", file=sys.stderr)
        return 1
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    if args.precision < 1:
        print(f"ERROR: --precision must be >= 1 (got {args.precision})",
              file=sys.stderr)
        return 1

    try:
        device = torch.device(args.device if args.device else
                              ("cuda" if torch.cuda.is_available() else "cpu"))
    except RuntimeError as exc:
        print(f"ERROR: invalid --device '{args.device}': {exc}", file=sys.stderr)
        return 1

    # Pre-scan record IDs so a duplicate aborts before any output is written.
    try:
        seen_scan: set[str] = set()
        opener = gzip.open if fasta_path.suffix == ".gz" else open
        with opener(fasta_path, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    tokens = line[1:].split()
                    if tokens and tokens[0] in seen_scan:
                        print(f"ERROR: duplicate FASTA record ID '{tokens[0]}' — "
                              f"IDs must be unique (they become the bedGraph "
                              f"chrom column).", file=sys.stderr)
                        return 1
                    if tokens:
                        seen_scan.add(tokens[0])
    except OSError as exc:
        print(f"ERROR: cannot read FASTA file: {exc}", file=sys.stderr)
        return 1

    try:
        model, max_seq_len = load_model(model_path, device, num_heads=args.num_heads)
    except Exception as exc:  # noqa: BLE001 — surface any load failure cleanly
        print(f"ERROR: failed to load model checkpoint: {exc}", file=sys.stderr)
        return 1

    if args.output == "-":
        out = sys.stdout
    else:
        try:
            out = open(args.output, "w")
        except OSError as exc:
            print(f"ERROR: cannot open output file: {exc}", file=sys.stderr)
            return 1
    # Provenance/column header as a '#' comment line: tolerated (skipped) by
    # bedtools/IGV and consistent with the SKIPPED/TRUNCATED marker lines.
    out.write(f"# predict_ribo_occ_fasta v{__version__} | model={model_path} | "
              f"window={3 * max_seq_len}nt | columns: transcript_id\tstart_0based\t"
              f"end_halfopen\toccupancy\n")
    t0 = time.time()
    n_records = n_predicted = n_skipped = total_rows = 0

    try:
        for record_id, raw_seq in read_fasta(fasta_path):
            n_records += 1
            seq = raw_seq.upper().replace("T", "U")
            n_other = sum(1 for c in seq if c not in "ACGU")
            if n_other:
                print(f"WARNING: {record_id}: {n_other} non-ACGT/U characters; "
                      f"codons containing them are treated as PAD tokens.",
                      file=sys.stderr)
            if len(seq) < 5:
                out.write(f"# SKIPPED\t{record_id}\tlength={len(seq)}nt\t"
                          f"reason=sequence_shorter_than_5nt\n")
                print(f"WARNING: {record_id}: sequence too short "
                      f"({len(seq)} nt < 5) — skipped.", file=sys.stderr)
                n_skipped += 1
                continue
            n_codons = len(seq) // 3
            window_nt = 3 * max_seq_len   # model window in nt, from checkpoint
            if n_codons > max_seq_len:
                if args.long_transcripts == "skip":
                    out.write(f"# SKIPPED\t{record_id}\tlength={len(seq)}nt\t"
                              f"reason=exceeds_model_window_{window_nt}nt\n")
                    print(f"WARNING: {record_id}: {len(seq)} nt exceeds the "
                          f"model window ({max_seq_len} codons = {window_nt} nt)"
                          f" — skipped.", file=sys.stderr)
                    n_skipped += 1
                    continue
                out.write(f"# TRUNCATED\t{record_id}\tlength={len(seq)}nt\t"
                          f"predicted=first_{window_nt}nt\t"
                          f"reason=exceeds_model_window_{window_nt}nt\n")
                print(f"WARNING: {record_id}: {len(seq)} nt exceeds the model "
                      f"window ({max_seq_len} codons = {window_nt} nt); "
                      f"only the 5' {window_nt} nt are predicted.",
                      file=sys.stderr)

            values = predict_interleaved(seq, model, max_seq_len, device)
            rows = write_bedgraph(out, record_id, values, precision=args.precision)
            total_rows += rows
            n_predicted += 1
            if not args.quiet:
                print(f"[predict_ribo_occ_fasta] {record_id}: {len(seq)} nt → "
                      f"{len(values)} positions, {rows} bedGraph rows",
                      file=sys.stderr)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # Downstream consumer (e.g. `| head`) closed the pipe: exit quietly.
        # Point stdout at /dev/null so the interpreter's shutdown flush
        # does not raise a second BrokenPipeError.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 141  # 128 + SIGPIPE, the standard shell convention
    finally:
        if out is not sys.stdout:
            out.close()

    if n_records == 0:
        print(f"ERROR: no FASTA records found in {fasta_path}", file=sys.stderr)
        return 1

    print(f"[predict_ribo_occ_fasta] done: {n_predicted}/{n_records} transcripts "
          f"predicted ({n_skipped} skipped), {total_rows} bedGraph rows, "
          f"{time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
