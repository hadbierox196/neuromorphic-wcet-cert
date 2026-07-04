#!/usr/bin/env bash
# End-to-end demo run on synthetic EEG data.
set -euo pipefail

python3 -m src.preprocessing --out artifacts/build/spike_tensors.npz

python3 -m src.train \
    --data artifacts/build/spike_tensors.npz \
    --model-out artifacts/build/best_model.pt \
    --report-out artifacts/build/training_report.json \
    --epochs 25 \
    --n-steps 8

python3 -m src.export_nir \
    --model artifacts/build/best_model.pt \
    --out artifacts/build/seizure_snn.nir.json

python3 -m src.compile_snn_mlir \
    --nir artifacts/build/seizure_snn.nir.json \
    --out artifacts/build/main.c

python3 -m src.certify \
    --c-source artifacts/build/main.c \
    --annotated-out artifacts/build/main.annotated.c \
    --lean-dir lean_stubs \
    --certificate-out artifacts/certificate.json

python3 -m src.iec62304_mapping --out artifacts/iec62304_coverage.json

echo ""
echo "Done. See artifacts/certificate.json and artifacts/iec62304_coverage.json"
