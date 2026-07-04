# neuromorphic-wcet-cert

Reference implementation of the pipeline described in *"Embedding Formal
Worst-Case Latency Proofs and Memory-Safety Certificates into the snn-mlir
MLIR Lowering Pipeline for IEC 62304-Compliant Edge Deployment of Spiking
Neural Networks."*

It takes a spiking neural network (SNN) for EEG seizure detection from
training through to a certification artifact bundle:

```
snnTorch model  →  NIR export  →  snn-mlir / C11 codegen
      →  IPET WCET analysis (ILP)  →  annotated C11 + Lean4 stubs
      →  IEC 62304 Class B artifact bundle (certificate.json)
```

## Status / honesty notice

This repo is a **research / demonstration pipeline**, not a certified medical
device toolchain. In particular:

- `snn-mlir` is an external, non-pip-installable research compiler. If it is
  not found on `PATH`, `src/compile_snn_mlir.py` automatically falls back to
  a small built-in C11 code generator that emits code in the same shape
  (`_mlir_ciface_snn_forward_step`, two persistent state buffers per layer)
  so the rest of the pipeline still runs end-to-end.
- WCET bounds are computed with a real IPET ILP formulation
  (`src/wcet_ipet.py`), but the ARM Cortex-M4F instruction timing table is a
  simplified model (see `CORTEX_M4F_TIMING` in that file), not a
  cycle-accurate hardware trace.
- Lean4 proof-obligation stubs are generated as `sorry`-terminated
  specifications. They are **not** discharged proofs.
- The IEC 62304 artifact mapping in `src/iec62304_mapping.py` documents
  *traceability*, not regulatory sign-off. §7.1 (risk management) is
  explicitly reported as uncovered, and this pipeline is not a substitute
  for it.
- CHB-MIT data access requires downloading from PhysioNet yourself; by
  default the pipeline runs on a small synthetic EEG generator so the whole
  thing works offline out of the box.

## Repository layout

```
neuromorphic-wcet-cert/
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # CHB-MIT / synthetic EEG -> spike tensors
│   ├── model.py             # 3-layer feedforward LIF SNN (snnTorch)
│   ├── train.py             # training loop + evaluation
│   ├── export_nir.py        # trained model -> NIR graph (JSON)
│   ├── compile_snn_mlir.py  # NIR -> C11, via snn-mlir or fallback codegen
│   ├── wcet_ipet.py         # CFG extraction + IPET ILP WCET solver
│   ├── certify.py           # orchestrator: annotation + Lean4 + certificate
│   └── iec62304_mapping.py  # IEC 62304 §5.x / §7.1 / §8.0 coverage table
├── lean_stubs/               # generated Lean4 proof-obligation stubs
├── artifacts/                 # generated certificate.json, build/, reports/
├── tests/
│   └── test_wcet_ipet.py
└── scripts/
    └── run_pipeline.sh        # end-to-end demo run
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# End-to-end demo on synthetic EEG data (no PhysioNet download required)
bash scripts/run_pipeline.sh
```

This will:

1. Generate/preprocess EEG spike tensors (`src/preprocessing.py`)
2. Train the 3-layer LIF SNN (`src/train.py`)
3. Export to NIR (`src/export_nir.py`)
4. Compile to C11 (`src/compile_snn_mlir.py`)
5. Run IPET WCET analysis and emit `@wcet_bound` annotations + Lean4 stubs
   + `artifacts/certificate.json` (`src/certify.py`)
6. Print the IEC 62304 coverage table (`src/iec62304_mapping.py`)

## Using real CHB-MIT data

Download patients `chb01` and `chb02` from
[PhysioNet](https://physionet.org/content/chbmit/1.0.0/) and point
`preprocessing.py` at the local directory:

```bash
python -m src.preprocessing --chbmit-dir /path/to/chbmit --patients chb01 chb02 \
    --out artifacts/build/spike_tensors.npz
```

## License

MIT — see `LICENSE`.
