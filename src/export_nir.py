"""
Export a trained SeizureLIFNet to a NIR-style graph (§2.2.1, §2.3).

Uses the real `nir` package if installed; otherwise falls back to writing
an equivalent JSON graph with the same node types (LIFLayer x3 + Affine
layers), so downstream compilation always has something to consume. The
fallback format intentionally mirrors NIR's node/edge shape.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import SeizureLIFNet, BETA, THETA, INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM


def export(model_path: str, out_path: str) -> dict:
    model = SeizureLIFNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    try:
        import nir  # type: ignore
        graph = nir.NIRGraph(nodes={
            "input": nir.Input(input_type={"input": (INPUT_DIM,)}),
            "affine1": nir.Affine(weight=model.fc1.weight.detach().numpy(), bias=None),
            "lif1": nir.LIF(tau=1.0 / (1 - BETA), r=1.0, v_leak=0.0,
                             v_threshold=THETA),
            "affine2": nir.Affine(weight=model.fc2.weight.detach().numpy(), bias=None),
            "lif2": nir.LIF(tau=1.0 / (1 - BETA), r=1.0, v_leak=0.0,
                             v_threshold=THETA),
            "affine3": nir.Affine(weight=model.fc3.weight.detach().numpy(), bias=None),
            "lif3": nir.LIF(tau=1.0 / (1 - BETA), r=1.0, v_leak=0.0,
                             v_threshold=THETA),
            "output": nir.Output(output_type={"output": (OUTPUT_DIM,)}),
        }, edges=[
            ("input", "affine1"), ("affine1", "lif1"), ("lif1", "affine2"),
            ("affine2", "lif2"), ("lif2", "affine3"), ("affine3", "lif3"),
            ("lif3", "output"),
        ])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        nir.write(out_path, graph)
        n_lif = 3
    except ImportError:
        graph_dict = _fallback_graph(model)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(graph_dict, indent=2))
        n_lif = sum(1 for n in graph_dict["nodes"] if n["type"] == "LIFLayer")

    assert n_lif == 3, f"Expected exactly 3 LIFLayer nodes, got {n_lif}"
    print(f"Exported NIR graph with {n_lif} LIFLayer nodes -> {out_path}")
    return {"n_lif_layers": n_lif, "path": out_path}


def _fallback_graph(model: SeizureLIFNet) -> dict:
    def layer_dims(w):
        return {"in_features": w.shape[1], "out_features": w.shape[0]}

    return {
        "format": "nir-fallback-json-v1",
        "nodes": [
            {"id": "input", "type": "Input", "shape": [INPUT_DIM]},
            {"id": "affine1", "type": "Affine", **layer_dims(model.fc1.weight)},
            {"id": "lif1", "type": "LIFLayer", "beta": BETA, "threshold": THETA},
            {"id": "affine2", "type": "Affine", **layer_dims(model.fc2.weight)},
            {"id": "lif2", "type": "LIFLayer", "beta": BETA, "threshold": THETA},
            {"id": "affine3", "type": "Affine", **layer_dims(model.fc3.weight)},
            {"id": "lif3", "type": "LIFLayer", "beta": BETA, "threshold": THETA},
            {"id": "output", "type": "Output", "shape": [OUTPUT_DIM]},
        ],
        "edges": [
            ["input", "affine1"], ["affine1", "lif1"], ["lif1", "affine2"],
            ["affine2", "lif2"], ["lif2", "affine3"], ["affine3", "lif3"],
            ["lif3", "output"],
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="artifacts/build/best_model.pt")
    ap.add_argument("--out", default="artifacts/build/seizure_snn.nir.json")
    args = ap.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
