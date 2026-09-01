# ribo_occ_predictor

Standalone FASTA → bedGraph ribosome-occupancy prediction with a trained
**RibosomeTranslator** model. Built for pipeline integration: single script,
no `aifofomo` checkout needed — only `torch` and `numpy`.

## What it does

For every transcript nucleotide sequence in the input FASTA, the model predicts
per-codon ribosome occupancy in all three reading frames directly from the
transformer head. The three frame tracks are interleaved into one
**per-nucleotide** track:

```
position 3k   <- frame_0[k]   (occupancy of the codon starting at nt 3k)
position 3k+1 <- frame_1[k]
position 3k+2 <- frame_2[k]
```

and written as bedGraph with **transcript-relative** coordinates
(0-based, half-open; the chrom column is the FASTA record ID):

```
ENST00000616016.5	0	1	0.0123
ENST00000616016.5	1	2	0.0110
ENST00000616016.5	2	3	0.4551
...
```

Consecutive positions with identical scores are merged into one interval
(standard bedGraph practice).

## Usage

```bash
python predict_ribo_occ_fasta.py transcripts.fasta \
    --model /path/to/model_500000.pt \
    -o occupancy.bedgraph

# or stream to stdout (all logging goes to stderr):
python predict_ribo_occ_fasta.py transcripts.fasta -m model.pt > occ.bedgraph
```

Options:

| Flag | Description |
|---|---|
| `-m, --model` | Path to the RibosomeTranslator `.pt` checkpoint (required) |
| `-o, --output` | Output bedGraph file; `-` = stdout (default) |
| `--device` | `cpu` / `cuda` / `mps` (default: cuda if available, else cpu) |
| `--long-transcripts` | `skip` (default) or `truncate` for transcripts longer than the model window |
| `--num-heads` | Attention heads; default inferred from checkpoint assuming head_dim=64 |
| `--precision` | Significant digits for scores (default 6) |
| `--quiet` | Suppress per-transcript progress on stderr |

## Input notes

* DNA (`T`) and RNA (`U`) both accepted; lowercase OK; `.gz` FASTA supported.
* Record ID = first whitespace-delimited token of the `>` header; must be unique.
* Codons containing non-ACGT/U characters (e.g. `N`) map to the PAD token but
  still receive a prediction.
* Transcripts longer than the model window are **skipped by default** and
  recorded as marker lines in the output (plus a stderr warning), e.g.:

  ```
  # SKIPPED	ENST00000589042.5	length=109224nt	reason=exceeds_model_window_12000nt
  ```

  With `--long-transcripts truncate` the 5' window is predicted instead and a
  `# TRUNCATED` marker line is written. Too-short records (< 5 nt) also get a
  `# SKIPPED` marker. The window is **derived from the checkpoint** at load
  time (`max_seq_len` codons from the positional-embedding shape × 3 nt; the
  current checkpoints give 4000 codons = 12,000 nt) — a future model trained
  with a larger window raises the limit automatically, including in the marker
  text. Marker lines start with `#` (ignored by bedtools/UCSC); strip them
  with `grep -v '^#'` if a strict parser objects.
* The final 2–4 nt of each transcript have no complete codon in every frame and
  get no bedGraph rows.

## Pipeline behavior

* stdout carries **only** bedGraph rows; all logging/warnings go to stderr.
* Exit codes: `0` success, `1` error (bad input/model/arguments, duplicate IDs
  — checked upfront, before any output is written), `141` downstream pipe
  closed early (e.g. `| head`), following the SIGPIPE convention.
* The model window assumption `head_dim=64` for inferring the head count is
  correct for the current checkpoints; if a future checkpoint uses a different
  head size, pass `--num-heads` explicitly (the load log flags when the value
  was inferred).

## Requirements

Python ≥ 3.10, `torch`, `numpy` (see `requirements.txt`).
