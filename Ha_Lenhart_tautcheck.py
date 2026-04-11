"""
Ha_Lenhart_tautcheck.py
───────────────────────
Tautology Checker for ESPRESSO PLA covers.
Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring.

Execution:
    python Ha_Lenhart_tautcheck.py <cover_file>

Output:
    - Prints whether the cover is a tautology.
    - If not, writes a witness (OFF-set cube) to <cover_file>_off_cube.
    - Prints instrumentation statistics to the terminal.

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) – HW3
"""

from __future__ import annotations
import sys
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from espresso_parser import Cover, parse_cover, write_cover, get_output_path


# ──────────────────────────────────────────────────────────────────────
# Instrumentation
# ──────────────────────────────────────────────────────────────────────

TAUTCHECK_TIMEOUT = 15 * 60  # 15 minutes in seconds


class TautologyTimeout(Exception):
    """Raised when the tautology check exceeds its time limit."""
    pass


@dataclass
class Stats:
    max_depth: int = 0
    base_case_tautology: int = 0     # tautology detected by base-case rules
    base_case_not_tautology: int = 0 # not-tautology detected by base-case rules
    unate_reductions: int = 0
    binate_splits: int = 0
    timed_out: bool = False


# ──────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────

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
        if c0 == 0:
            continue
        c1 = col.count('1')
        if c1 == 0:
            continue
            
        literal_count = c0 + c1
        balance = abs(c1 - c0)
        if (literal_count > best_literal_count or
                (literal_count == best_literal_count and balance < best_balance)):
            best_var = v
            best_literal_count = literal_count
            best_balance = balance

    return best_var

# ──────────────────────────────────────────────────────────────────────
# Main recursive tautology checker
# ──────────────────────────────────────────────────────────────────────

def _tautology_check(
    cubes: List[str],
    num_vars: int,
    stats: Stats,
    depth: int,
    deadline: float,
    iterations: List[int]
) -> Tuple[bool, Optional[str]]:
    
    stats.max_depth = max(stats.max_depth, depth)

    iterations[0] += 1
    if (iterations[0] & 1023) == 0 and deadline and time.perf_counter() > deadline:
        raise TautologyTimeout()

    if not cubes:
        stats.base_case_not_tautology += 1
        return False, '0' * num_vars

    if ('-' * num_vars) in cubes:
        stats.base_case_tautology += 1
        return True, None

    cols = list(zip(*cubes))

    for v in range(num_vars):
        col = cols[v]
        if '1' not in col and '-' not in col:
            stats.base_case_not_tautology += 1
            w = list('0' * num_vars)
            w[v] = '1'
            return False, ''.join(w)
        if '0' not in col and '-' not in col:
            stats.base_case_not_tautology += 1
            w = list('0' * num_vars)
            w[v] = '0'
            return False, ''.join(w)

    # ── Unate check ───────────────────────────────────────────────────
    # If the entire cover is unate and lacks the universal cube,
    # it is NOT a tautology.
    is_unate = True
    for col in cols:
        if '0' in col and '1' in col:
            is_unate = False
            break

    if is_unate:
        stats.unate_reductions += 1
        stats.base_case_not_tautology += 1
        witness = []
        for v in range(num_vars):
            col = cols[v]
            if '1' in col and '0' not in col:
                witness.append('0')
            elif '0' in col and '1' not in col:
                witness.append('1')
            else:
                witness.append('0')
        return False, ''.join(witness)

    # ── Unate variable cofactoring ────────────────────────────────────
    # If variable x is unate, we only need ONE cofactor instead of two:
    #   Positive-unate (only 1/-): F≡1 iff F_x̄ ≡ 1
    #   Negative-unate (only 0/-): F≡1 iff F_x  ≡ 1
    # This halves the recursion tree at each unate variable.
    for v in range(num_vars):
        col = cols[v]
        has_0 = '0' in col
        has_1 = '1' in col
        if has_1 and not has_0:
            # Positive unate: only need negative cofactor
            stats.unate_reductions += 1
            neg_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '1']
            is_taut, witness = _tautology_check(neg_cubes, num_vars, stats, depth + 1, deadline, iterations)
            if not is_taut:
                w = list(witness)
                w[v] = '0'
                return False, ''.join(w)
            return True, None
        elif has_0 and not has_1:
            # Negative unate: only need positive cofactor
            stats.unate_reductions += 1
            pos_cubes = [c[:v] + '-' + c[v + 1:] for c in cubes if c[v] != '0']
            is_taut, witness = _tautology_check(pos_cubes, num_vars, stats, depth + 1, deadline, iterations)
            if not is_taut:
                w = list(witness)
                w[v] = '1'
                return False, ''.join(w)
            return True, None

    # ── Binate split (Shannon expansion) ──────────────────────────────
    stats.binate_splits += 1

    var = _pick_binate_variable(cols, num_vars)

    # Positive cofactor (var = 1)
    pos_cubes = [c[:var] + '-' + c[var + 1:] for c in cubes if c[var] != '0']
    is_taut_pos, witness_pos = _tautology_check(pos_cubes, num_vars, stats, depth + 1, deadline, iterations)
    if not is_taut_pos:
        w = list(witness_pos)
        w[var] = '1'
        return False, ''.join(w)

    # Negative cofactor (var = 0)
    neg_cubes = [c[:var] + '-' + c[var + 1:] for c in cubes if c[var] != '1']
    is_taut_neg, witness_neg = _tautology_check(neg_cubes, num_vars, stats, depth + 1, deadline, iterations)
    if not is_taut_neg:
        w = list(witness_neg)
        w[var] = '0'
        return False, ''.join(w)

    return True, None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def check_tautology(
    cover: Cover,
    timeout: float = TAUTCHECK_TIMEOUT,
) -> Tuple[bool, Optional[str], Stats]:
    
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


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_tautcheck.py <path_to_file_or_folder> [<path2> ...]")
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
        report.append(f"  Tautology Check: {os.path.basename(filepath)}")
        report.append(f"{'='*60}")

        cover = parse_cover(filepath)
        report.append(f"  Variables : {cover.num_inputs}")
        report.append(f"  Cubes     : {cover.num_cubes}")

        # Start measurement
        tracemalloc.start()
        t_start = time.perf_counter()

        is_taut, witness, stats = check_tautology(cover)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t_end - t_start

        # Results
        if stats.timed_out:
            report.append(f"  Result    : TIMEOUT (exceeded {TAUTCHECK_TIMEOUT // 60} min)")
        elif is_taut:
            report.append(f"  Result    : TAUTOLOGY")
        else:
            report.append(f"  Result    : NOT a tautology")
            report.append(f"  Witness   : {witness}")

            # Write witness file
            witness_path = get_output_path(filepath, "_off_cube", "Tautcheck-Results")
            witness_cover = Cover(
                num_inputs=cover.num_inputs,
                num_outputs=cover.num_outputs,
                input_labels=cover.input_labels,
                output_labels=cover.output_labels,
                cubes=[witness],
            )
            write_cover(witness_cover, witness_path)
            report.append(f"  Witness written to: {witness_path}")

        # Instrumentation
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

        # Print to terminal
        for line in report:
            print(line)

        # Write report to text file in results folder
        report_path = get_output_path(filepath, "_tautcheck_report.txt", "Tautcheck-Reports")
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"\n  Report written to: {report_path}")


if __name__ == "__main__":
    main()
