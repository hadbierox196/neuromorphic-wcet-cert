"""
IPET (Implicit Path Enumeration Technique) WCET analysis (§2.4).

For each LIF layer's step function, the CFG is a single loop (over output
neurons, each with an inner loop over input weights) plus a handful of
straight-line blocks. We build the standard IPET ILP:

    maximize   sum(c_i * e_i)                over CFG edges i
    subject to flow-conservation at every node
               e_loop_back <= T_steps          (loop bound)
               e_i >= 0

and solve it two ways for cross-validation, as in the paper: once with
`scipy.optimize.linprog` and once with `pulp`.

Instruction timing is a simplified ARM Cortex-M4F model (Table 2 values).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
import pulp

# ARM Cortex-M4F TRM r0p1 simplified per-instruction cycle costs (Table 2)
CORTEX_M4F_TIMING = {
    "FMAC": 2,   # 1 + 1 stall
    "FCMP": 1,
    "FSTS": 2,
    "LDR": 2,
    "BRANCH": 1,
    "LOOP_OVERHEAD": 2,  # increment + compare + branch, amortized
}


@dataclass
class CFGEdge:
    src: str
    dst: str
    cost: int
    is_loop_back: bool = False


@dataclass
class LIFLayerCFG:
    name: str
    n_in: int
    n_out: int
    edges: list[CFGEdge] = field(default_factory=list)


def extract_cfg(c_source: str) -> list[LIFLayerCFG]:
    """
    Regex-based CFG extraction from the fallback/compiled C11 output
    (§2.5 step 1). Recovers n_in/n_out per layer from the declared static
    buffer/weight sizes and builds a two-level-loop CFG:

        entry -> outer_loop_header
        outer_loop_header -> inner_loop_header      (enter inner loop)
        inner_loop_header -> inner_loop_header       (loop-back, n_in times)
        inner_loop_header -> outer_loop_footer       (exit inner loop)
        outer_loop_footer -> outer_loop_header       (loop-back, n_out times)
        outer_loop_footer -> exit
    """
    layers = []
    pattern = re.compile(
        r"void (lif_layer\d+)_step\(.*?\)\s*\{.*?for \(int i = 0; i < (\d+); i\+\+\)"
        r".*?for \(int j = 0; j < (\d+); j\+\+\)",
        re.DOTALL,
    )
    for match in pattern.finditer(c_source):
        name, n_out, n_in = match.group(1), int(match.group(2)), int(match.group(3))
        layers.append(_build_layer_cfg(name, n_in, n_out))
    return layers


def _build_layer_cfg(name: str, n_in: int, n_out: int) -> LIFLayerCFG:
    t = CORTEX_M4F_TIMING
    inner_body_cost = t["LDR"] + t["FMAC"]                       # acc += w[i,j]*in[j]
    inner_loop_cost = inner_body_cost + t["LOOP_OVERHEAD"]
    outer_body_cost = (t["FCMP"] + t["FSTS"] + t["LOOP_OVERHEAD"])  # spike compare + store + reset
    outer_loop_cost = outer_body_cost

    edges = [
        CFGEdge("entry", "outer_header", t["LOOP_OVERHEAD"]),
        CFGEdge("outer_header", "inner_header", t["BRANCH"]),
        CFGEdge("inner_header", "inner_header", inner_loop_cost, is_loop_back=True),
        CFGEdge("inner_header", "outer_footer", outer_body_cost),
        CFGEdge("outer_footer", "outer_header", outer_loop_cost, is_loop_back=True),
        CFGEdge("outer_footer", "exit", t["BRANCH"]),
    ]
    return LIFLayerCFG(name=name, n_in=n_in, n_out=n_out, edges=edges)


def solve_ipet_scipy(layer: LIFLayerCFG) -> float:
    """Solve the IPET ILP (relaxed to LP, which is exact for this structure
    since the polytope is integral for a single-loop CFG) with scipy."""
    edges = layer.edges
    n = len(edges)
    costs = np.array([e.cost for e in edges], dtype=float)

    nodes = sorted({e.src for e in edges} | {e.dst for e in edges})
    A_eq, b_eq = [], []
    for node in nodes:
        if node in ("entry", "exit"):
            continue
        row = np.zeros(n)
        for k, e in enumerate(edges):
            if e.dst == node:
                row[k] += 1
            if e.src == node:
                row[k] -= 1
        A_eq.append(row)
        b_eq.append(0.0)

    # Loop bounds: inner loop-back <= n_in per outer iteration i.e. total
    # inner executions <= n_in * n_out; outer loop-back <= n_out - 1.
    A_ub, b_ub = [], []
    for k, e in enumerate(edges):
        if e.is_loop_back and e.src == "inner_header":
            row = np.zeros(n); row[k] = 1
            A_ub.append(row); b_ub.append(layer.n_in * layer.n_out)
        if e.is_loop_back and e.src == "outer_footer":
            row = np.zeros(n); row[k] = 1
            A_ub.append(row); b_ub.append(max(layer.n_out - 1, 0))

    # entry edge flow == 1 (single execution of the function)
    entry_row = np.zeros(n)
    for k, e in enumerate(edges):
        if e.src == "entry":
            entry_row[k] = 1
    A_eq.append(entry_row); b_eq.append(1.0)

    res = linprog(
        c=-costs,  # maximize -> minimize negative
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=[(0, None)] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"IPET LP infeasible for layer {layer.name}: {res.message}")
    return float(-res.fun)


def solve_ipet_pulp(layer: LIFLayerCFG) -> float:
    """Cross-validation solve with PuLP (§2.4)."""
    prob = pulp.LpProblem(f"ipet_{layer.name}", pulp.LpMaximize)
    e = {i: pulp.LpVariable(f"e{i}", lowBound=0) for i in range(len(layer.edges))}

    prob += pulp.lpSum(layer.edges[i].cost * e[i] for i in e)

    nodes = sorted({edge.src for edge in layer.edges} | {edge.dst for edge in layer.edges})
    for node in nodes:
        if node in ("entry", "exit"):
            continue
        inflow = pulp.lpSum(e[i] for i, edge in enumerate(layer.edges) if edge.dst == node)
        outflow = pulp.lpSum(e[i] for i, edge in enumerate(layer.edges) if edge.src == node)
        prob += inflow == outflow

    for i, edge in enumerate(layer.edges):
        if edge.is_loop_back and edge.src == "inner_header":
            prob += e[i] <= layer.n_in * layer.n_out
        if edge.is_loop_back and edge.src == "outer_footer":
            prob += e[i] <= max(layer.n_out - 1, 0)

    prob += pulp.lpSum(e[i] for i, edge in enumerate(layer.edges) if edge.src == "entry") == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return float(pulp.value(prob.objective))


def analyze(c_source: str, uniform_bound: bool = True) -> dict:
    """
    Run IPET on every LIF layer in the C11 source. Per §3.3/§4.1, the paper
    applies a uniform per-layer bound equal to the largest layer's WCET as a
    conservative simplification; set uniform_bound=False for tighter,
    individualized per-layer bounds (§4.5 P1).
    """
    layers = extract_cfg(c_source)
    if not layers:
        raise ValueError("No LIF layer step functions found in C11 source")

    per_layer = {}
    for layer in layers:
        scipy_bound = solve_ipet_scipy(layer)
        pulp_bound = solve_ipet_pulp(layer)
        if abs(scipy_bound - pulp_bound) > 1e-3:
            raise RuntimeError(
                f"scipy/pulp IPET disagreement for {layer.name}: "
                f"{scipy_bound} vs {pulp_bound}"
            )
        per_layer[layer.name] = int(round(scipy_bound))

    if uniform_bound:
        bound = max(per_layer.values())
        applied = {name: bound for name in per_layer}
    else:
        applied = per_layer

    network_bound = sum(applied.values())
    return {
        "per_layer_raw": per_layer,
        "per_layer_applied": applied,
        "network_bound_cycles": network_bound,
        "uniform_bound": uniform_bound,
        "n_flow_variables": sum(len(l.edges) for l in layers[:1]),
        "n_constraints": None,  # filled in by certify.py from solver metadata if needed
    }
