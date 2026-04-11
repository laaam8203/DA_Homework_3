"""
Ha_Lenhart_complgen.py
──────────────────────
Complement Cover Generator for ESPRESSO PLA covers.
Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring
to compute the complement in SCC-minimal form.

Execution:
    python Ha_Lenhart_complgen.py <cover_file> [<cover_file2> ...]

Output:
    - Writes complement cover to <cover_file>_compl in ESPRESSO format.
    - Prints execution statistics to the terminal.

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
import time
import tracemalloc
from typing import List, Tuple

from espresso_parser import Cover, parse_cover, write_cover, get_output_path


COMPLGEN_TIMEOUT = 60 * 60  # 1 hour in seconds


class ComplementTimeout(Exception):
    """Raised when complement generation exceeds its time limit."""
    pass


# ======================================================================
# Core helpers (shared with tautology checker logic)
# ======================================================================

def _has_all_dc_row(cubes: List[str], num_vars: int) -> bool:
    return ('-' * num_vars) in cubes

def _get_columns(cubes: List[str]) -> List[Tuple[str, ...]]:
    return list(zip(*cubes))

def _is_unate_fast(cols: List[Tuple[str, ...]]) -> bool:
    for col in cols:
        if '0' in col and '1' in col:
            return False
    return True

def _cofactor(cubes: List[str], var: int, val: str) -> List[str]:
    opp = '1' if val == '0' else '0'
    return [c[:var] + '-' + c[var + 1:] for c in cubes if c[var] != opp]

def _pick_binate_variable(cols: List[Tuple[str, ...]], num_vars: int) -> int:
    best_var = -1
    best_literal_count = -1
    best_balance = float('inf')

    for v in range(num_vars):
        col = cols[v]
        c0 = col.count('0')
        if c0 == 0: continue
        c1 = col.count('1')
        if c1 == 0: continue
        
        literal_count = c0 + c1
        balance = abs(c1 - c0)
        if (literal_count > best_literal_count or
                (literal_count == best_literal_count and balance < best_balance)):
            best_var = v
            best_literal_count = literal_count
            best_balance = balance

    return best_var

# ======================================================================
# AND / OR of cube lists  (intersection / union helpers)
# ======================================================================

def _and_cube(a: str, b: str) -> str | None:
    result = []
    for ca, cb in zip(a, b):
        if ca == '-': result.append(cb)
        elif cb == '-': result.append(ca)
        elif ca == cb: result.append(ca)
        else: return None
    return ''.join(result)

def _and_cube_with_literal(cube: str, var: int, val: str) -> str | None:
    ch = cube[var]
    if ch == val or ch == '-':
        return cube[:var] + val + cube[var + 1:]
    return None

# ======================================================================
# Complement via URP   (F_bar = x * Fx_bar  +  x' * Fx'_bar)
# ======================================================================

def _complement(cubes: List[str], num_vars: int, deadline: float, iterations: List[int]) -> List[str]:
    iterations[0] += 1
    if (iterations[0] & 511) == 0 and deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    if not cubes:
        return ['-' * num_vars]

    if ('-' * num_vars) in cubes:
        return []

    if len(cubes) == 1:
        return _complement_single_cube(cubes[0], num_vars)

    cols = list(zip(*cubes))
    is_unate = True
    for col in cols:
        if '0' in col and '1' in col:
            is_unate = False
            break

    if is_unate:
        return _complement_unate(cubes, num_vars, cols, deadline, iterations)

    var = _pick_binate_variable(cols, num_vars)

    pos_cubes = [c[:var] + '-' + c[var + 1:] for c in cubes if c[var] != '0']
    neg_cubes = [c[:var] + '-' + c[var + 1:] for c in cubes if c[var] != '1']

    compl_pos = _complement(pos_cubes, num_vars, deadline, iterations)
    compl_neg = _complement(neg_cubes, num_vars, deadline, iterations)

    result = []
    for c in compl_pos:
        result.append(c[:var] + '1' + c[var + 1:])
    for c in compl_neg:
        result.append(c[:var] + '0' + c[var + 1:])

    return _merge_adjacent(result, num_vars)


def _complement_single_cube(cube: str, num_vars: int) -> List[str]:
    result = []
    for i, ch in enumerate(cube):
        if ch == '-': continue
        new_cube = ['-'] * num_vars
        new_cube[i] = '1' if ch == '0' else '0'
        result.append(''.join(new_cube))
    return result


def _complement_unate(cubes: List[str], num_vars: int, cols: List[Tuple[str, ...]], deadline: float, iterations: List[int]) -> List[str]:
    for v in range(num_vars):
        col = cols[v]
        c0 = '0' in col
        c1 = '1' in col
        if c0 or c1:
            if c1:
                pos_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '0']
                neg_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '1']
                compl_pos = _complement(pos_cubes, num_vars, deadline, iterations)
                compl_neg = _complement(neg_cubes, num_vars, deadline, iterations)
                result = []
                for c in compl_pos:
                    result.append(c[:v] + '1' + c[v + 1:])
                for c in compl_neg:
                    result.append(c[:v] + '0' + c[v + 1:])
                return _merge_adjacent(result, num_vars)
            else:
                pos_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '0']
                neg_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '1']
                compl_pos = _complement(pos_cubes, num_vars, deadline, iterations)
                compl_neg = _complement(neg_cubes, num_vars, deadline, iterations)
                result = []
                for c in compl_pos:
                    result.append(c[:v] + '1' + c[v + 1:])
                for c in compl_neg:
                    result.append(c[:v] + '0' + c[v + 1:])
                return _merge_adjacent(result, num_vars)
    return []


# ======================================================================
# Simple SCC-minimality helper: merge adjacent cubes
# ======================================================================

def _merge_adjacent(cubes: List[str], num_vars: int) -> List[str]:
    changed = True
    while changed:
        changed = False
        new_cubes = []
        n = len(cubes)
        used = [False] * n
        for i in range(n):
            if used[i]: continue
            merged = False
            a = cubes[i]
            for j in range(i + 1, n):
                if used[j]: continue
                b = cubes[j]
                
                diff_count = 0
                diff_pos = -1
                for k in range(num_vars):
                    if a[k] != b[k]:
                        diff_count += 1
                        diff_pos = k
                        if diff_count > 1: break
                
                if diff_count == 1 and a[diff_pos] in '01' and b[diff_pos] in '01':
                    new_cubes.append(a[:diff_pos] + '-' + a[diff_pos + 1:])
                    used[i] = True
                    used[j] = True
                    merged = True
                    changed = True
                    break
            
            if not merged:
                new_cubes.append(a)
        cubes = new_cubes

    return _remove_contained(cubes, num_vars)


def _remove_contained(cubes: List[str], num_vars: int) -> List[str]:
    keep = []
    n = len(cubes)
    for i in range(n):
        ci = cubes[i]
        contained = False
        for j in range(n):
            if i == j: continue
            cj = cubes[j]
            
            is_sub = True
            for k in range(num_vars):
                if cj[k] != '-' and cj[k] != ci[k]:
                    is_sub = False
                    break
            
            if is_sub:
                if ci == cj and j > i:
                    continue
                contained = True
                break
                
        if not contained:
            keep.append(ci)
    return keep


# ======================================================================
# Public API
# ======================================================================

def generate_complement(cover: Cover, timeout: float = COMPLGEN_TIMEOUT) -> Cover:
    """Compute and return the complement of *cover* as a new Cover.
    Returns a Cover with an empty cube list if the operation times out.
    """
    deadline = time.perf_counter() + timeout if timeout else 0
    iterations = [0]
    
    try:
        compl_cubes = _complement(cover.cubes, cover.num_inputs, deadline, iterations)
        timed_out = False
    except ComplementTimeout:
        compl_cubes = []
        timed_out = True
        
    result = Cover(
        num_inputs=cover.num_inputs,
        num_outputs=cover.num_outputs,
        input_labels=cover.input_labels,
        output_labels=cover.output_labels,
        cubes=compl_cubes,
    )
    result._timed_out = timed_out
    return result


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <path_to_file_or_folder> [<path2> ...]")
        sys.exit(1)

    targets = []
    for arg_path in sys.argv[1:]:
        if os.path.isdir(arg_path):
            for file in os.listdir(arg_path):
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
        report.append(f"  Complement Generation: {os.path.basename(filepath)}")
        report.append(f"{'='*60}")

        cover = parse_cover(filepath)
        report.append(f"  Variables    : {cover.num_inputs}")
        report.append(f"  Input cubes  : {cover.num_cubes}")

        tracemalloc.start()
        t_start = time.perf_counter()

        compl = generate_complement(cover)
        timed_out = getattr(compl, '_timed_out', False)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t_end - t_start

        if timed_out:
            report.append(f"  Result       : TIMEOUT (exceeded {COMPLGEN_TIMEOUT // 60} min)")
        else:
            # Write complement file
            out_path = get_output_path(filepath, "_compl", "Complgen-Results")
            write_cover(compl, out_path)
            report.append(f"  Output cubes : {compl.num_cubes}")
            report.append(f"  Written to   : {out_path}")

        # Stats
        report.append(f"")
        report.append(f"  -- Instrumentation --")
        report.append(f"  Execution time : {elapsed:.6f} s")
        report.append(f"  Peak memory    : {peak_mem / 1024:.2f} KB")
        if timed_out:
            report.append(f"  Status         : TIMED OUT")

        # Print to terminal
        for line in report:
            print(line)

        # Write report to text file in results folder
        report_path = get_output_path(filepath, "_complgen_report.txt", "Complgen-Reports")
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\n  Report written to: {report_path}")


if __name__ == "__main__":
    main()
