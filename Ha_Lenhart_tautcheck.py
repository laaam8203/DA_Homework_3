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

@dataclass
class Stats:
    max_depth: int = 0
    base_case_tautology: int = 0     # tautology detected by base-case rules
    base_case_not_tautology: int = 0 # not-tautology detected by base-case rules
    unate_reductions: int = 0
    binate_splits: int = 0


# ──────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────

def _column(cubes: List[str], var: int) -> Tuple[int, int, int]:
    """Return (count_0, count_1, count_dc) for variable *var*."""
    c0 = c1 = cd = 0
    for cube in cubes:
        ch = cube[var]
        if ch == '0':
            c0 += 1
        elif ch == '1':
            c1 += 1
        else:
            cd += 1
    return c0, c1, cd


def _is_unate(cubes: List[str], num_vars: int) -> bool:
    """A cover is unate if every variable appears in only one polarity
    (positive or negative) across all cubes – don't-cares are allowed."""
    for v in range(num_vars):
        c0, c1, _ = _column(cubes, v)
        if c0 > 0 and c1 > 0:
            return False
    return True


def _has_all_dc_row(cubes: List[str]) -> bool:
    """True if some cube is all don't-cares (covers the entire space)."""
    return any(all(ch == '-' for ch in c) for c in cubes)


def _cofactor(cubes: List[str], var: int, val: str) -> List[str]:
    """
    Cofactor the cube list with respect to variable *var* = *val*.
    val is '0' or '1'.
    - Cubes with the *opposite* literal in that column are removed.
    - Remaining cubes have that column replaced by '-'.
    """
    opp = '1' if val == '0' else '0'
    result = []
    for cube in cubes:
        ch = cube[var]
        if ch == opp:
            continue  # cube disappears
        # Replace column with '-'
        new_cube = cube[:var] + '-' + cube[var + 1:]
        result.append(new_cube)
    return result


def _pick_binate_variable(cubes: List[str], num_vars: int) -> int:
    """
    Choose the most binate variable for splitting.
    Heuristic: pick the binate variable whose column has the most
    literals (i.e. fewest don't-cares → most constraining).
    Among ties, pick the one closest to balanced (|c1 - c0| minimal).
    """
    best_var = -1
    best_literal_count = -1
    best_balance = float('inf')

    for v in range(num_vars):
        c0, c1, cd = _column(cubes, v)
        if c0 == 0 or c1 == 0:
            continue  # unate variable – skip
        literal_count = c0 + c1
        balance = abs(c1 - c0)
        if (literal_count > best_literal_count or
                (literal_count == best_literal_count and balance < best_balance)):
            best_var = v
            best_literal_count = literal_count
            best_balance = balance

    return best_var


# ──────────────────────────────────────────────────────────────────────
# Unate tautology check (base case for unate covers)
# ──────────────────────────────────────────────────────────────────────

def _unate_tautology(cubes: List[str], num_vars: int) -> bool:
    """
    For a *unate* cover the tautology check is simple:
    the cover is a tautology iff it contains a cube that is all
    don't-cares (i.e. covers the whole space).

    More precisely, a unate cover F is a tautology iff there exists a
    cube that has '-' in every *essential* column.  But the simplest
    correct check is: the cover contains a row of all '-'.
    """
    return _has_all_dc_row(cubes)


# ──────────────────────────────────────────────────────────────────────
# Main recursive tautology checker
# ──────────────────────────────────────────────────────────────────────

def _tautology_check(
    cubes: List[str],
    num_vars: int,
    stats: Stats,
    depth: int,
) -> Tuple[bool, Optional[str]]:
    """
    Recursive tautology check using URP.

    Returns
    -------
    (is_tautology, witness_or_None)
        If is_tautology is False, *witness* is a minterm string of
        length num_vars (all '0'/'1') that is NOT covered.
    """
    stats.max_depth = max(stats.max_depth, depth)

    # ── Base cases ────────────────────────────────────────────────────

    # 1. Empty cover → not a tautology.  Any point is a witness.
    if not cubes:
        stats.base_case_not_tautology += 1
        return False, '0' * num_vars

    # 2. If any cube is all don't-cares → tautology.
    if _has_all_dc_row(cubes):
        stats.base_case_tautology += 1
        return True, None

    # 3. If a column is all '0' or all '1' (no don't-care, no opposite),
    #    that means every cube restricts that variable to one value,
    #    so the opposite value is never covered → not a tautology.
    for v in range(num_vars):
        c0, c1, cd = _column(cubes, v)
        if c1 == 0 and cd == 0:
            # Every cube has '0' in column v → value '1' is uncovered
            stats.base_case_not_tautology += 1
            witness = list('0' * num_vars)
            witness[v] = '1'
            return False, ''.join(witness)
        if c0 == 0 and cd == 0:
            # Every cube has '1' in column v → value '0' is uncovered
            stats.base_case_not_tautology += 1
            witness = list('0' * num_vars)
            witness[v] = '0'
            return False, ''.join(witness)

    # ── Unate check ───────────────────────────────────────────────────
    if _is_unate(cubes, num_vars):
        stats.unate_reductions += 1
        if _unate_tautology(cubes, num_vars):
            stats.base_case_tautology += 1
            return True, None
        else:
            stats.base_case_not_tautology += 1
            # Build witness: for each variable pick the value that does
            # NOT appear (the "missing" polarity).  If a variable only
            # appears as '1' (positive unate), then '0' is missing and
            # we set witness to '0'.  If only '0' or '-', set witness='0'
            # as a safe default.
            witness = []
            for v in range(num_vars):
                c0, c1, cd = _column(cubes, v)
                if c1 > 0 and c0 == 0:
                    witness.append('0')  # positive unate → '0' not covered
                elif c0 > 0 and c1 == 0:
                    witness.append('1')  # negative unate → '1' not covered
                else:
                    witness.append('0')  # all dc
            return False, ''.join(witness)

    # ── Binate split (Shannon expansion) ──────────────────────────────
    var = _pick_binate_variable(cubes, num_vars)
    stats.binate_splits += 1

    # Positive cofactor (var = 1)
    pos_cubes = _cofactor(cubes, var, '1')
    is_taut_pos, witness_pos = _tautology_check(pos_cubes, num_vars, stats, depth + 1)
    if not is_taut_pos:
        # Witness from positive cofactor: set var = 1
        w = list(witness_pos)
        w[var] = '1'
        return False, ''.join(w)

    # Negative cofactor (var = 0)
    neg_cubes = _cofactor(cubes, var, '0')
    is_taut_neg, witness_neg = _tautology_check(neg_cubes, num_vars, stats, depth + 1)
    if not is_taut_neg:
        # Witness from negative cofactor: set var = 0
        w = list(witness_neg)
        w[var] = '0'
        return False, ''.join(w)

    return True, None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def check_tautology(
    cover: Cover,
) -> Tuple[bool, Optional[str], Stats]:
    """
    Check whether *cover* is a tautology.

    Returns
    -------
    (is_tautology, witness_cube_or_None, stats)
    """
    stats = Stats()
    is_taut, witness = _tautology_check(
        cover.cubes, cover.num_inputs, stats, depth=0
    )
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
        if is_taut:
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
