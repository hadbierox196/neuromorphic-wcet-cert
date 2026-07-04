"""
IEC 62304 Class B artifact coverage table (§2.6, §3.5, Table 3).

Coverage scoring: 1.0 full, 0.5 partial, 0.0 none. Score is reported only
over the eight §5.x development-process clauses; §7.1 (risk management) and
§8.0 (configuration management) are tracked separately as supplementary
categories, matching the paper's credit scheme.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

COVERAGE_SCORES = {"full": 1.0, "partial": 0.5, "none": 0.0}


@dataclass
class ClauseCoverage:
    clause: str
    requirement: str
    artifact: str
    coverage: str  # "full" | "partial" | "none"


DEFAULT_TABLE: list[ClauseCoverage] = [
    ClauseCoverage("5.1", "Software development planning",
                    "environment_check.txt, run scripts / notebooks", "full"),
    ClauseCoverage("5.2", "Software requirements analysis",
                    "research log, gap analysis (this README + paper draft)", "full"),
    ClauseCoverage("5.3", "Software architectural design",
                    "CFG documentation (wcet_ipet.py docstrings), NIR graph JSON", "full"),
    ClauseCoverage("5.4", "Software detailed design",
                    "annotated C11 source, annotation notes (certify.py)", "full"),
    ClauseCoverage("5.5", "Software unit implementation",
                    "compiled C11 output (main.c / main.annotated.c)", "full"),
    ClauseCoverage("5.6", "Software integration & testing",
                    "Python vs. C11 divergence test (not yet implemented here)", "partial"),
    ClauseCoverage("5.7", "Software system testing",
                    "end-to-end pipeline test on held-out windows (tests/)", "partial"),
    ClauseCoverage("5.8", "Software release",
                    "repository not yet published under a permanent DOI", "none"),
    ClauseCoverage("7.1", "Software risk management",
                    "not performed — explicit limitation, see README", "none"),
    ClauseCoverage("8.0", "Software configuration management",
                    "version-pinned requirements.txt, pinned commit/LLVM version", "full"),
]

SECTION_5X_CLAUSES = {"5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8"}


def score(table: list[ClauseCoverage] = DEFAULT_TABLE) -> dict:
    section5 = [c for c in table if c.clause in SECTION_5X_CLAUSES]
    total = sum(COVERAGE_SCORES[c.coverage] for c in section5)
    max_total = float(len(section5))
    return {
        "section_5x_score": total,
        "section_5x_max": max_total,
        "section_5x_fraction": total / max_total if max_total else 0.0,
        "clauses": [c.__dict__ for c in table],
    }


def print_table(table: list[ClauseCoverage] = DEFAULT_TABLE) -> None:
    result = score(table)
    print(f"{'Clause':<8}{'Coverage':<10}{'Requirement'}")
    print("-" * 60)
    for c in table:
        print(f"{c.clause:<8}{c.coverage:<10}{c.requirement}")
    print("-" * 60)
    print(f"Section 5.x score: {result['section_5x_score']:.1f} / "
          f"{result['section_5x_max']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/iec62304_coverage.json")
    args = ap.parse_args()
    result = score()
    print_table()
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
