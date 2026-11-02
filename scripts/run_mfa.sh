#!/usr/bin/env bash
# Train an MFA acoustic model on one language's selected corpus and write TextGrids.
# Uses a private micromamba environment, so Colab's Python and PyTorch are untouched.
# Usage: bash scripts/run_mfa.sh english|telugu
set -euo pipefail

LANGUAGE="$1"
DATA_ROOT="${FS2_DATA_ROOT:-data}"
MFA_ROOT="${FS2_MFA_ROOT:-/content/mfa}"
INTERIM="$DATA_ROOT/interim/$LANGUAGE"
export MAMBA_ROOT_PREFIX="$MFA_ROOT/mamba"
export MFA_ROOT_DIR="$MFA_ROOT/runtime"

if [ ! -x "$MFA_ROOT/bin/micromamba" ]; then
  mkdir -p "$MFA_ROOT"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MFA_ROOT" bin/micromamba
fi
MAMBA="$MFA_ROOT/bin/micromamba"
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/aligner" ]; then
  "$MAMBA" create -y -n aligner -c conda-forge montreal-forced-aligner
fi

"$MAMBA" run -n aligner mfa version
"$MAMBA" run -n aligner mfa validate --clean -j "$(nproc)" --single_speaker \
  "$INTERIM/mfa_corpus" "$INTERIM/lexicon.txt" > "$INTERIM/mfa_validate.log" 2>&1 || \
  echo "Validation reported problems; see $INTERIM/mfa_validate.log"
# English needs a three-frame minimum per phone: the default collapses TH, G and R to one frame.
TOPOLOGY=()
if [ "$LANGUAGE" = "english" ]; then
  TOPOLOGY=(--topology_path English/mfa_topology.yaml)
fi
"$MAMBA" run -n aligner mfa train --clean -j "$(nproc)" --single_speaker "${TOPOLOGY[@]}" \
  --output_directory "$INTERIM/textgrids" \
  "$INTERIM/mfa_corpus" "$INTERIM/lexicon.txt" "$INTERIM/acoustic_model.zip"
# mfa train exports the final alignments, so no separate mfa align pass is run.
find "$INTERIM/textgrids" -name '*.TextGrid' | wc -l
