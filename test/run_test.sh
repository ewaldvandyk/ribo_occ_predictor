#!/usr/bin/env bash
# Run the FASTA→bedGraph predictor on the CDS test set and validate the output.
# The predicted bedGraph is kept for inspection (default: test_output.bedgraph
# next to this script).
#
# Usage:  ./run_test.sh /path/to/model.pt [python] [output.bedgraph]
set -euo pipefail
MODEL="${1:?usage: run_test.sh /path/to/model.pt [python] [output.bedgraph]}"
PYTHON="${2:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${3:-$HERE/test_output.bedgraph}"

"$PYTHON" "$HERE/../predict_ribo_occ_fasta.py" "$HERE/test_transcripts.fasta" \
    -m "$MODEL" -o "$OUT" --quiet
"$PYTHON" "$HERE/check_output.py" "$OUT" "$HERE/expected_cds.tsv"
echo "bedGraph kept at: $OUT"
