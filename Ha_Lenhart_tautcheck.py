"""
Ha_Lenhart_tautcheck.py
───────────────────────
Tautology Checker for ESPRESSO PLA covers.
Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring.
Encoding: '0'→1, '1'→2, '-'→3 (uint8 arrays).

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) – HW3
"""

from __future__ import annotations
import sys
import os
import time
import tracemalloc
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from espresso_parser import Cover, parse_cover, write_cover, get_output_path

TAUTCHECK_TIMEOUT = 15 * 60

class TautologyTimeout(Exception):
    pass

@dataclass
class Stats:
    """Recursive execution counters for instrumentation reporting."""
    max_depth: int = 0
    base_case_tautology: int = 0
    base_case_not_tautology: int = 0
    unate_reductions: int = 0
    binate_splits: int = 0
    timed_out: bool = False

# ──────────────────────────────────────────────────────────────────────
# Core URP Recursion
# ──────────────────────────────────────────────────────────────────────
def _tautology_check(
    cubes: np.ndarray,
    num_vars: int,
    stats: Stats,
    depth: int,
    deadline: float,
    iterations: List[int]
) -> Tuple[bool, Optional[str]]:
    """Recursively determine if `cubes` is a tautology.
    Returns (is_tautology, witness_string_or_None)."""

    stats.max_depth = max(stats.max_depth, depth)
    iterations[0] += 1
    if (iterations[0] & 1023) == 0 and deadline and time.perf_counter() > deadline:
        raise TautologyTimeout()

    # Base case: empty cover is not a tautology
    if len(cubes) == 0:
        stats.base_case_not_tautology += 1
        return False, '0' * num_vars

    # Base case: universal cube ('-'*n) present → tautology
    if np.any(np.all(cubes == 3, axis=1)):
        stats.base_case_tautology += 1
        return True, None

    c0 = np.sum(cubes == 1, axis=0)
    c1 = np.sum(cubes == 2, axis=0)
    cd = np.sum(cubes == 3, axis=0)

    # Column shortcuts: a column with no coverage in one polarity is a trivial gap
    missing_1_dc = (c1 == 0) & (cd == 0)
    if np.any(missing_1_dc):
        stats.base_case_not_tautology += 1
        v = np.argmax(missing_1_dc)
        w = list('0' * num_vars); w[v] = '1'
        return False, ''.join(w)

    missing_0_dc = (c0 == 0) & (cd == 0)
    if np.any(missing_0_dc):
        stats.base_case_not_tautology += 1
        v = np.argmax(missing_0_dc)
        w = list('0' * num_vars); w[v] = '0'
        return False, ''.join(w)

    has_0 = c0 > 0
    has_1 = c1 > 0

    # Entirely unate cover: not a tautology (no universal cube was found above)
    unate_vars = ~(has_0 & has_1)
    if np.all(unate_vars):
        stats.unate_reductions += 1
        stats.base_case_not_tautology += 1
        witness = []
        for v in range(num_vars):
            if has_1[v] and not has_0[v]: witness.append('0')
            elif has_0[v] and not has_1[v]: witness.append('1')
            else: witness.append('0')
        return False, ''.join(witness)

    # Multi-variable unate reduction: cofactor all unate vars simultaneously.
    # For pos-unate x_j: check x_j=0 subspace (exclude cubes requiring x_j=1).
    # For neg-unate x_j: check x_j=1 subspace (exclude cubes requiring x_j=0).
    strictly_unate = unate_vars & (has_0 | has_1)
    if np.any(strictly_unate):
        stats.unate_reductions += 1
        indices = np.where(strictly_unate)[0]
        mask = np.ones(len(cubes), dtype=bool)
        for v in indices:
            mask &= (cubes[:, v] != (2 if has_1[v] else 1))
        new_cubes = cubes[mask].copy()
        new_cubes[:, indices] = 3
        if len(new_cubes) > 1000 and (depth % 5 == 0):  # periodic dedup
            new_cubes = np.unique(new_cubes, axis=0)
        is_taut, witness = _tautology_check(new_cubes, num_vars, stats, depth + 1, deadline, iterations)
        if not is_taut:
            w = list(witness)
            for v in indices:
                w[v] = '0' if has_1[v] else '1'  # reconstruct witness
            return False, ''.join(w)
        return True, None

    # Binate split: max literal count primary, min balance secondary
    stats.binate_splits += 1
    binate_mask = has_0 & has_1
    lit_counts = c0 + c1
    balances = np.abs(c1 - c0)
    valid_lit = np.where(binate_mask, lit_counts, -1)
    candidates = (valid_lit == np.max(valid_lit))
    var = np.argmin(np.where(candidates, balances, float('inf')))

    pos_cubes = cubes[cubes[:, var] != 1].copy(); pos_cubes[:, var] = 3
    is_taut_pos, witness_pos = _tautology_check(pos_cubes, num_vars, stats, depth + 1, deadline, iterations)
    if not is_taut_pos:
        w = list(witness_pos); w[var] = '1'
        return False, ''.join(w)

    neg_cubes = cubes[cubes[:, var] != 2].copy(); neg_cubes[:, var] = 3
    is_taut_neg, witness_neg = _tautology_check(neg_cubes, num_vars, stats, depth + 1, deadline, iterations)
    if not is_taut_neg:
        w = list(witness_neg); w[var] = '0'
        return False, ''.join(w)

    return True, None

# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def check_tautology(
    cover: Cover,
    timeout: float = TAUTCHECK_TIMEOUT,
) -> Tuple[bool, Optional[str], Stats]:
    """Check if `cover` is a tautology. Returns (is_taut, witness, stats)."""
    stats = Stats()
    deadline = time.perf_counter() + timeout if timeout else 0
    iterations = [0]
    try:
        is_taut, witness = _tautology_check(
            cover.cubes, cover.num_inputs, stats, depth=0, deadline=deadline, iterations=iterations
        )
    except TautologyTimeout:
        stats.timed_out = True
        return False, None, stats
    return is_taut, witness, stats


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_tautcheck.py <path_to_file_or_folder> [<path2> ...]")
        sys.exit(1)

    targets = []
    for arg_path in sys.argv[1:]:
        if os.path.isdir(arg_path):
            files = sorted(os.listdir(arg_path))
            for file in files:
                full_path = os.path.join(arg_path, file)
                if os.path.isfile(full_path) and not file.startswith('.'):
                    targets.append(full_path)
        elif os.path.isfile(arg_path):
            targets.append(arg_path)
        else:
            print(f"Skipping invalid path: {arg_path}")

    for filepath in targets:
        report = []
        report.append(f"{'='*60}")
        report.append(f"  Tautology Check: {os.path.basename(filepath)}")
        report.append(f"{'='*60}")

        cover = parse_cover(filepath)
        report.append(f"  Variables : {cover.num_inputs}")
        report.append(f"  Cubes     : {cover.num_cubes}")

        tracemalloc.start()
        t_start = time.perf_counter()

        is_taut, witness, stats = check_tautology(cover)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t_end - t_start

        if stats.timed_out:
            report.append(f"  Result    : TIMEOUT (exceeded {TAUTCHECK_TIMEOUT // 60} min)")
        elif is_taut:
            report.append(f"  Result    : TAUTOLOGY")
        else:
            report.append(f"  Result    : NOT a tautology")
            report.append(f"  Witness   : {witness}")

            witness_path = get_output_path(filepath, "_off_cube", "Tautcheck-Results")
            witness_cover = Cover(
                num_inputs=cover.num_inputs,
                num_outputs=cover.num_outputs,
                input_labels=cover.input_labels,
                output_labels=cover.output_labels,
                cubes=np.array([[3 if ch=='-' else 2 if ch=='1' else 1 for ch in witness]], dtype=np.uint8),
            )
            write_cover(witness_cover, witness_path)
            report.append(f"  Witness written to: {witness_path}")

        report.append(f"")
        report.append(f"  -- Instrumentation --")
        report.append(f"  Execution time     : {elapsed:.6f} s")
        report.append(f"  Peak memory        : {peak_mem / 1024:.2f} KB")
        report.append(f"  Max recursion depth : {stats.max_depth}")
        report.append(f"  Base-case tautology : {stats.base_case_tautology}")
        report.append(f"  Base-case NOT taut  : {stats.base_case_not_tautology}")
        report.append(f"  Unate reductions    : {stats.unate_reductions}")
        report.append(f"  Binate splits       : {stats.binate_splits}")
        if stats.timed_out:
            report.append(f"  Status             : TIMED OUT")

        for line in report:
            print(line)

        report_path = get_output_path(filepath, "_tautcheck_report.txt", "Tautcheck-Reports")
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\n  Report written to: {report_path}")

if __name__ == "__main__":
    main()
