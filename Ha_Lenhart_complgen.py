"""
Ha_Lenhart_complgen.py
───────────────────────
Complement Generation for ESPRESSO PLA covers.
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
from typing import List

from espresso_parser import Cover, parse_cover, write_cover, get_output_path

COMPLGEN_TIMEOUT = 60 * 60  

class ComplementTimeout(Exception):
    pass

# ======================================================================
# Cleanup: SCC-minimal reduction of the complement cover
# ======================================================================

def _remove_contained(cubes: np.ndarray, deadline: float) -> np.ndarray:
    """Remove any cube subsumed by another: Ci ⊆ Cj iff (Ci | Cj) == Cj."""
    n = len(cubes)
    if n <= 1: return cubes
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if (i & 2047) == 0 and deadline and time.perf_counter() > deadline:
            raise ComplementTimeout()
        if not keep[i]: continue
        ci = cubes[i]
        contained_in_other = np.all((ci | cubes) == cubes, axis=1)
        contained_in_other[i] = False
        if np.any(contained_in_other):
            keep[i] = False
    return cubes[keep]

def _merge_adjacent(cubes: np.ndarray, deadline: float) -> np.ndarray:
    """Iteratively merge distance-1 cube pairs (differ in exactly one variable)."""
    if len(cubes) <= 1: return cubes
    changed = True
    while changed:
        changed = False
        n = len(cubes)
        new_cubes = []
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            if (i & 1023) == 0 and deadline and time.perf_counter() > deadline:
                raise ComplementTimeout()
            if used[i]: continue
            ci = cubes[i]
            # XOR detects dist-1 pairs: exactly one position differs (val 1^2=3), rest match
            diff = ci ^ cubes[i+1:]
            match = (np.sum(diff == 3, axis=1) == 1) & (np.sum(diff == 0, axis=1) == (cubes.shape[1] - 1))
            if np.any(match):
                j = i + 1 + int(np.argmax(match))
                if not used[j]:
                    new_cubes.append(ci | cubes[j])
                    used[i] = True
                    used[j] = True
                    changed = True
                    continue
            new_cubes.append(ci)
        if changed:
            cubes = np.array(new_cubes, dtype=np.uint8)
    return _remove_contained(cubes, deadline)

# ======================================================================
# Core URP Recursion
# ======================================================================

def _complement_single_cube(cube: np.ndarray, num_vars: int) -> np.ndarray:
    """Shannon expansion base case: complement of a single cube.
    Each literal l_j becomes its own cube with polarity flipped (3-val)."""
    valid_vars = np.where(cube != 3)[0]
    n_res = len(valid_vars)
    if n_res == 0: return np.empty((0, num_vars), dtype=np.uint8)
    res = np.full((n_res, num_vars), 3, dtype=np.uint8)
    res[np.arange(n_res), valid_vars] = 3 - cube[valid_vars]  # flip polarity
    return res

def _complement(cubes: np.ndarray, num_vars: int, deadline: float, iterations: List[int], depth: int) -> np.ndarray:
    iterations[0] += 1
    if (iterations[0] & 2047) == 0 and deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    # Base cases
    if len(cubes) == 0:                          # empty cover → universal complement
        return np.full((1, num_vars), 3, dtype=np.uint8)
    if np.any(np.all(cubes == 3, axis=1)):       # tautology cube → empty complement
        return np.empty((0, num_vars), dtype=np.uint8)
    if len(cubes) == 1:
        return _complement_single_cube(cubes[0], num_vars)

    # Common cube extraction: F = C·F_rest → F_bar = C_bar + C·F_rest_bar
    common = np.bitwise_or.reduce(cubes, axis=0)
    if np.any(common != 3):
        compl_common = _complement_single_cube(common, num_vars)
        rest_cubes = cubes | (common ^ 3)        # expand cubes to remove common literals
        compl_rest = _complement(rest_cubes, num_vars, deadline, iterations, depth + 1)
        if len(compl_rest) > 0:
            compl_rest[:, common != 3] = common[common != 3]  # restrict result to common literals
            return np.vstack([compl_common, compl_rest])
        return compl_common

    c0 = np.sum(cubes == 1, axis=0)
    c1 = np.sum(cubes == 2, axis=0)
    has_0 = c0 > 0
    has_1 = c1 > 0

    # ── Unate variable optimization (lecture theorem) ─────────────────────
    # For pos-unate x_j (appears only as x=1 or dc, never x=0):
    #   F̄ = complement(F_{x_j=1})        ← x_j left as dc, spans BOTH halves
    #       + x̄_j · complement(F_{x_j=0}) ← extra holes specific to x_j=0
    # Correctness: by pos-unate monotonicity F(x=0) ⊆ F(x=1), so any x=1 hole
    # is also an x=0 hole. The first term's cubes (with x_j=dc) are therefore
    # valid across both half-spaces without needing a literal. Only the x=0-
    # exclusive holes need the x̄_j literal in the second term.
    # Neg-unate is symmetric (swap roles of x=0 and x=1).
    # Advantage over naive binate split: keeps x_j=dc in the first-term cubes,
    # producing a more compact cover (larger cubes cover more minterms).
    unate_mask = (~(has_0 & has_1)) & (has_0 | has_1)
    if np.any(unate_mask):
        u_indices = np.where(unate_mask)[0]
        vi = int(u_indices[np.argmax((c0 + c1)[u_indices])])  # most-literal unate var
        pos_unate = bool(has_1[vi])   # True → pos-unate, False → neg-unate

        if pos_unate:
            # F_{vi=1}: all cubes, vi→dc (larger cofactor)
            # F_{vi=0}: cubes where vi≠2, vi→dc (smaller cofactor)
            f_large = cubes.copy();               f_large[:, vi] = 3
            f_small = cubes[cubes[:, vi] != 2].copy(); f_small[:, vi] = 3
            compl_large = _complement(f_large, num_vars, deadline, iterations, depth + 1)
            # compl_large: x_j stays dc — valid for both halves by monotonicity ✓
            compl_small = _complement(f_small, num_vars, deadline, iterations, depth + 1)
            if len(compl_small) > 0:
                compl_small[:, vi] = 1          # pin to x̄_j (value 1 = '0')
        else:
            # Neg-unate: F_{vi=0} is the larger cofactor (symmetric)
            f_large = cubes.copy();               f_large[:, vi] = 3
            f_small = cubes[cubes[:, vi] != 1].copy(); f_small[:, vi] = 3
            compl_large = _complement(f_large, num_vars, deadline, iterations, depth + 1)
            # compl_large: x_j stays dc — valid for both halves ✓
            compl_small = _complement(f_small, num_vars, deadline, iterations, depth + 1)
            if len(compl_small) > 0:
                compl_small[:, vi] = 2          # pin to x_j (value 2 = '1')

        res_list = [r for r in [compl_large, compl_small] if len(r) > 0]
        return np.vstack(res_list) if res_list else np.empty((0, num_vars), dtype=np.uint8)

    # ── Binate split ──────────────────────────────────────────────────────
    # All remaining variables are binate (both polarities). If we reach here,
    # binate_mask is guaranteed to be non-empty (otherwise all cubes would be
    # all-dc, caught by the universal-cube base case above).
    # Variable selection: primary = most literals (most constrained), 
    #                     secondary = most balanced (minimize |c0-c1|)
    # Balancing the two branches reduces asymmetric recursion depth.
    binate_mask = has_0 & has_1
    lit_counts  = c0 + c1
    balances    = np.abs(c1 - c0)
    valid_lit   = np.where(binate_mask, lit_counts, -1)
    candidates  = (valid_lit == np.max(valid_lit))        # all vars tied for max literals
    var = int(np.argmin(np.where(candidates, balances, np.iinfo(np.int64).max)))

    pos_cubes = cubes[cubes[:, var] != 1].copy(); pos_cubes[:, var] = 3
    neg_cubes = cubes[cubes[:, var] != 2].copy(); neg_cubes[:, var] = 3

    compl_pos = _complement(pos_cubes, num_vars, deadline, iterations, depth + 1)
    compl_neg = _complement(neg_cubes, num_vars, deadline, iterations, depth + 1)

    res_list = []
    if len(compl_pos) > 0: compl_pos[:, var] = 2; res_list.append(compl_pos)
    if len(compl_neg) > 0: compl_neg[:, var] = 1; res_list.append(compl_neg)
    return np.vstack(res_list) if res_list else np.empty((0, num_vars), dtype=np.uint8)

# ======================================================================
# Public API
# ======================================================================
def generate_complement(cover: Cover, timeout: float = COMPLGEN_TIMEOUT) -> Cover:
    """Compute the complement of a cover using URP. Returns (cover, timed_out, iterations)."""
    deadline = time.perf_counter() + timeout if timeout else 0
    iterations = [0]
    try:
        compl_int = _complement(cover.cubes, cover.num_inputs, deadline, iterations, 0)
        if len(compl_int) > 1:                   # final SCC-minimal cleanup
            compl_int = np.unique(compl_int, axis=0)
            compl_int = _merge_adjacent(compl_int, deadline)
        timed_out = False
    except ComplementTimeout:
        compl_int = np.empty((0, cover.num_inputs), dtype=np.uint8)
        timed_out = True
        
    return Cover(
        num_inputs=cover.num_inputs,
        num_outputs=cover.num_outputs,
        input_labels=cover.input_labels,
        output_labels=cover.output_labels,
        cubes=compl_int,
    ), timed_out, iterations[0]

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <path_to_file_or_folder>")
        sys.exit(1)

    targets = []
    for arg_path in sys.argv[1:]:
        if os.path.exists(arg_path):
            if os.path.isdir(arg_path):
                files = os.listdir(arg_path)
                for file in sorted(files):
                    fpath = os.path.join(arg_path, file)
                    if os.path.isfile(fpath) and not file.startswith("."):
                        targets.append(fpath)
            else:
                targets.append(arg_path)

    for filepath in targets:
        report = []
        report.append(f"{'='*60}")
        report.append(f"  Complement Generation: {os.path.basename(filepath)}")
        report.append(f"{'='*60}")

        cover = parse_cover(filepath)
        report.append(f"  Variables    : {cover.num_inputs}")
        report.append(f"  Input cubes  : {cover.num_cubes}")

        tracemalloc.start()
        t_start = time.perf_counter()

        compl_cover, timed_out, iterations = generate_complement(cover)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t_end - t_start

        if timed_out:
            report.append(f"  Result       : TIMEOUT (exceeded {COMPLGEN_TIMEOUT // 60} min)")
            report.append(f"  Output cubes : {len(compl_cover.cubes)} (Partial/Invalid)")
        else:
            report.append(f"  Output cubes : {len(compl_cover.cubes)}")

            out_path = get_output_path(filepath, "_compl", "Complgen-Results")
            write_cover(compl_cover, out_path)
            report.append(f"  Written to   : {out_path}")

        report.append(f"")
        report.append(f"  -- Instrumentation --")
        report.append(f"  Execution time : {elapsed:.6f} s")
        report.append(f"  Peak memory    : {peak_mem / 1024:.2f} KB")
        report.append(f"")

        for line in report:
            print(line)

        report_path = get_output_path(filepath, "_complgen_report.txt", "Complgen-Reports")
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"  Report written to: {report_path}")

if __name__ == "__main__":
    main()
