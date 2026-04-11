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
# Bitmask helpers
# ──────────────────────────────────────────────────────────────────────

def _str_to_masks(s: str, n: int) -> Tuple[int, int]:
    m1 = 0; m0 = 0
    for ch in s:
        m1 <<= 1; m0 <<= 1
        if ch == '1':   m1 |= 1
        elif ch == '0': m0 |= 1
        else:           m1 |= 1; m0 |= 1
    return m1, m0

def _masks_to_str(m1: int, m0: int, n: int) -> str:
    chars = []
    for i in range(n):
        shift = n - 1 - i
        b1 = (m1 >> shift) & 1
        b0 = (m0 >> shift) & 1
        if b1 and b0:   chars.append('-')
        elif b1:        chars.append('1')
        elif b0:        chars.append('0')
        else:           chars.append('0')  # Fallback valid state
    return ''.join(chars)

def _remove_contained_cubes(cubes: List[Tuple[int, int]], all_ones: int) -> List[Tuple[int, int]]:
    """Remove cubes that are fully subsumed (contained) by another cube."""
    n = len(cubes)
    if n <= 1: return cubes
    
    processed = []
    for on, off in cubes:
        care = (on ^ off) & all_ones
        processed.append((care.bit_count(), care, on, off))
        
    # Sort placing largest cubes (fewest care bits) first to maximize early subsumption
    processed.sort(key=lambda x: x[0])
    
    keep = []
    for bc_i, care_i, on_i, off_i in processed:
        is_subsumed = False
        # Only need to check against larger cubes already in keep
        for care_j, on_j, off_j in keep:
            if (care_j & ~care_i) == 0 and (on_j & care_j) == (on_i & care_j):
                is_subsumed = True
                break
        if not is_subsumed:
            keep.append((care_i, on_i, off_i))
            
    return [(on, off) for care, on, off in keep]


# ──────────────────────────────────────────────────────────────────────
# Main recursive tautology checker
# ──────────────────────────────────────────────────────────────────────

def _tautology_check(
    cubes: List[Tuple[int, int]],
    active_vars: int,
    num_vars: int,
    all_ones: int,
    stats: Stats,
    depth: int,
    deadline: float = 0,
) -> Tuple[bool, int, int]:
    stats.max_depth = max(stats.max_depth, depth)

    if not cubes:
        stats.base_case_not_tautology += 1
        return False, 0, all_ones  # minterm with all 0s

    for on, off in cubes:
        if on == all_ones and off == all_ones:
            stats.base_case_tautology += 1
            return True, 0, 0

    if active_vars == 0:
        stats.base_case_not_tautology += 1
        return False, 0, all_ones

    cubes = _remove_contained_cubes(cubes, all_ones)

    # Recheck all-dc within active vars
    for on, off in cubes:
        care = (on ^ off) & all_ones
        if care & active_vars == 0:
            stats.base_case_tautology += 1
            return True, 0, 0

    best_var = -1
    best_score = -1
    best_balance = float('inf')
    is_unate = True
    n_c = len(cubes)

    for v in range(num_vars):
        vbit = 1 << (num_vars - 1 - v)
        if not (active_vars & vbit): continue
        c1 = sum(1 for on, off in cubes if (on & vbit) and not (off & vbit))
        c0 = sum(1 for on, off in cubes if (off & vbit) and not (on & vbit))
        cd = n_c - c1 - c0

        if c1 == 0 and cd == 0:
            stats.base_case_not_tautology += 1
            return False, vbit, all_ones & ~vbit
        if c0 == 0 and cd == 0:
            stats.base_case_not_tautology += 1
            return False, all_ones & ~vbit, vbit

        if c0 > 0 and c1 > 0:
            is_unate = False
            score = c0 + c1
            bal = abs(c1 - c0)
            if score > best_score or (score == best_score and bal < best_balance):
                best_var = v; best_score = score; best_balance = bal

    if is_unate:
        stats.unate_reductions += 1
        stats.base_case_not_tautology += 1
        w1=0; w0=0
        for v in range(num_vars):
            vbit = 1 << (num_vars - 1 - v)
            if not (active_vars & vbit):
                w0 |= vbit; continue
            c1 = any((on & vbit) and not (off & vbit) for on, off in cubes)
            c0 = any((off & vbit) and not (on & vbit) for on, off in cubes)
            if c1 and not c0: w0 |= vbit
            elif c0 and not c1: w1 |= vbit
            else: w0 |= vbit
        return False, w1, w0

    if deadline and time.perf_counter() > deadline:
        raise TautologyTimeout()

    assert best_var >= 0
    stats.binate_splits += 1
    vbit = 1 << (num_vars - 1 - best_var)
    new_active = active_vars & ~vbit

    # pos cofactor (vbit = 1) -> remove if cube needs 0
    pos_cubes = list(set((on | vbit, off | vbit) for on, off in cubes if not ((off & vbit) and not (on & vbit))))
    is_taut_p, wp1, wp0 = _tautology_check(pos_cubes, new_active, num_vars, all_ones, stats, depth + 1, deadline)
    if not is_taut_p:
        wp1 |= vbit; wp0 &= ~vbit
        return False, wp1, wp0

    neg_cubes = list(set((on | vbit, off | vbit) for on, off in cubes if not ((on & vbit) and not (off & vbit))))
    is_taut_n, wn1, wn0 = _tautology_check(neg_cubes, new_active, num_vars, all_ones, stats, depth + 1, deadline)
    if not is_taut_n:
        wn1 &= ~vbit; wn0 |= vbit
        return False, wn1, wn0

    return True, 0, 0


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def check_tautology(
    cover: Cover,
    timeout: float = TAUTCHECK_TIMEOUT,
) -> Tuple[bool, Optional[str], Stats]:
    """
    Check whether *cover* is a tautology.
    """
    stats = Stats()
    num_vars = cover.num_inputs
    all_ones = (1 << num_vars) - 1
    mask_cubes = [_str_to_masks(c, num_vars) for c in cover.cubes]
    deadline = time.perf_counter() + timeout if timeout else 0
    try:
        is_taut, w1, w0 = _tautology_check(
            mask_cubes, all_ones, num_vars, all_ones, stats, 0, deadline
        )
        witness = _masks_to_str(w1, w0, num_vars) if not is_taut else None
    except TautologyTimeout:
        stats.timed_out = True
        return False, None, stats
    return is_taut, witness, stats


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_tautcheck.py <cover_file> [<cover_file2> ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
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
