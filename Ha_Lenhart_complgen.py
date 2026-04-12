"""
Ha_Lenhart_complgen.py
───────────────────────
Complement Generation for ESPRESSO PLA covers.
Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring.
Converted to Numpy Vectorized Boolean Logic Engine.

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
# Vectorized Natively C-Compiled Cleanup System
# ======================================================================

def _remove_contained(cubes: np.ndarray) -> np.ndarray:
    n = len(cubes)
    if n <= 1: return cubes
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]: continue
        ci = cubes[i]
        
        # Ci contained in Cj iff (Ci | Cj) == Cj. Vectorized across all Cjs simultaneously
        contains_ci = np.all((ci | cubes) == cubes, axis=1)
        contains_ci[i] = False
        identicals = np.all(cubes == ci, axis=1)
        contains_ci[identicals] = False
        
        if np.any(contains_ci):
            keep[i] = False
    return cubes[keep]

def _merge_adjacent(cubes: np.ndarray) -> np.ndarray:
    if len(cubes) <= 1: return cubes
    changed = True
    while changed:
        changed = False
        n = len(cubes)
        new_cubes = []
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            if used[i]: continue
            ci = cubes[i]
            
            # XOR logic: strictly one field equals `3` (1^2=3), all others equal 0 matching
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
            
    return _remove_contained(cubes)

# ======================================================================
# Vectorized Core Cofactoring Rules
# ======================================================================

def _complement_single_cube(cube: np.ndarray, num_vars: int) -> np.ndarray:
    valid_vars = np.where(cube != 3)[0]
    n_res = len(valid_vars)
    if n_res == 0: return np.empty((0, num_vars), dtype=np.uint8)
    
    res = np.full((n_res, num_vars), 3, dtype=np.uint8)
    for i, v in enumerate(valid_vars):
        res[i, v] = 3 - cube[v]
    return res

def _complement(cubes: np.ndarray, num_vars: int, deadline: float, iterations: List[int], depth: int) -> np.ndarray:
    iterations[0] += 1
    if (iterations[0] & 511) == 0 and deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    if len(cubes) == 0:
        return np.full((1, num_vars), 3, dtype=np.uint8)

    if np.any(np.all(cubes == 3, axis=1)):
        return np.empty((0, num_vars), dtype=np.uint8)

    if len(cubes) == 1:
        return _complement_single_cube(cubes[0], num_vars)
        
    common = np.bitwise_or.reduce(cubes, axis=0) # Extracted literally via C-OR
    if np.any(common != 3):
        flip_mask = (common ^ 3)
        f_common = cubes | flip_mask
        compl_f_common = _complement(f_common, num_vars, deadline, iterations, depth + 1)
        res1 = _complement_single_cube(common, num_vars)
        if len(compl_f_common) > 0:
            res = np.vstack([res1, compl_f_common])
            res = np.unique(res, axis=0)
            if len(res) > 2000:
                res = _merge_adjacent(res)
            return res
        return res1

    c0 = np.sum(cubes == 1, axis=0)
    c1 = np.sum(cubes == 2, axis=0)

    has_0 = c0 > 0
    has_1 = c1 > 0

    unate_vars = ~(has_0 & has_1)
    if np.all(unate_vars):
        # Pick first unate variable
        v = int(np.where(has_0 | has_1)[0][0])
        f1 = cubes.copy()
        f1[:, v] = 3
        if has_1[v]: # Positive Unate 
            f0 = cubes[cubes[:, v] == 3].copy()
            compl_f0 = _complement(f0, num_vars, deadline, iterations, depth + 1)
            compl_f1 = _complement(f1, num_vars, deadline, iterations, depth + 1)
            if len(compl_f0) > 0:
                compl_f0[:, v] = 1 # '0'
                compl_f1 = np.vstack([compl_f0, compl_f1])
        else: # Negative Unate
            f0 = cubes[cubes[:, v] == 3].copy() # actually f1 but functionally f0 equivalent mapping
            compl_f1 = _complement(f0, num_vars, deadline, iterations, depth + 1)
            compl_f0 = _complement(f1, num_vars, deadline, iterations, depth + 1)
            if len(compl_f1) > 0:
                compl_f1[:, v] = 2 # '1'
                compl_f1 = np.vstack([compl_f1, compl_f0])
        
        result = np.unique(compl_f1, axis=0)
        if len(result) > 1000:
            result = _merge_adjacent(result)
        return result

    # Standard Binate Split
    binate_mask = has_0 & has_1
    lit_counts = c0 + c1
    balances = np.abs(c1 - c0)
    
    valid_lit = np.where(binate_mask, lit_counts, -1)
    max_lit = np.max(valid_lit)
    candidates = (valid_lit == max_lit)
    
    valid_balances = np.where(candidates, balances, float('inf'))
    var = int(np.argmin(valid_balances))

    pos_cubes = cubes[cubes[:, var] != 1].copy()
    pos_cubes[:, var] = 3

    neg_cubes = cubes[cubes[:, var] != 2].copy()
    neg_cubes[:, var] = 3
    
    compl_pos = _complement(pos_cubes, num_vars, deadline, iterations, depth + 1)
    compl_neg = _complement(neg_cubes, num_vars, deadline, iterations, depth + 1)

    res_list = []
    if len(compl_pos) > 0:
        compl_pos[:, var] = 2
        res_list.append(compl_pos)
    if len(compl_neg) > 0:
        compl_neg[:, var] = 1
        res_list.append(compl_neg)
        
    if res_list:
        result = np.vstack(res_list)
        result = np.unique(result, axis=0)
        if len(result) > 1000:
            result = _merge_adjacent(result)
        return result
    return np.empty((0, num_vars), dtype=np.uint8)

# ======================================================================
# Wrapper 
# ======================================================================
def generate_complement(cover: Cover, timeout: float = COMPLGEN_TIMEOUT) -> Cover:
    deadline = time.perf_counter() + timeout if timeout else 0
    iterations = [0]
    
    num_vars = cover.num_inputs
    try:
        compl_int = _complement(cover.cubes, num_vars, deadline, iterations, 0)
        compl_int = _merge_adjacent(compl_int)
        timed_out = False
    except ComplementTimeout:
        compl_int = np.empty((0, num_vars), dtype=np.uint8)
        timed_out = True
        
    return Cover(
        num_inputs=cover.num_inputs,
        num_outputs=cover.num_outputs,
        input_labels=cover.input_labels,
        output_labels=cover.output_labels,
        cubes=compl_int,
    ), timed_out, iterations[0]

# ======================================================================
# Setup and Entry
# ======================================================================
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <path_to_file_or_folder> [<path2> ...]")
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

    print(f"\nDiscovered {len(targets)} generic test benches for Complgen.\n")

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
