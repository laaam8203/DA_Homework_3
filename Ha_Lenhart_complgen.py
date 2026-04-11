"""
Ha_Lenhart_complgen.py
──────────────────────
Complement Generator for ESPRESSO PLA covers.
Uses the Unate Recursive Paradigm (URP) with Shannon expansion to compute
the exact complement (OFF-set).

Execution:
    python Ha_Lenhart_complgen.py <cover_file>

Output:
    - Writes the complement cover to <cover_file>_compl
    - Prints instrumentation statistics to the terminal.

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) – HW3
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
# Bitmask helpers
# ======================================================================

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
        else:           chars.append('0')
    return ''.join(chars)


def _merge_adjacent(cubes: List[Tuple[int, int]], all_ones: int) -> List[Tuple[int, int]]:
    """
    Merge distance-1 cubes to keep the complement size minimal.
    Distance-1: Two cubes agree everywhere except in one bit where one has 0 and another has 1.
    They can be merged to '-' (both bits 1).
    """
    if len(cubes) < 2:
        return cubes

    merged_any = True
    while merged_any:
        merged_any = False
        n = len(cubes)
        keep = [True] * n
        new_cubes = []
        for i in range(n):
            if not keep[i]: continue
            on_i, off_i = cubes[i]
            merged_this = False
            for j in range(i + 1, n):
                if not keep[j]: continue
                on_j, off_j = cubes[j]
                
                # Check distance 1:
                # They must agree on care vs dc structure (optional but simplifies)
                # Actually, simpler: they differ in exactly one literal.
                diff_on = on_i ^ on_j
                diff_off = off_i ^ off_j
                # To be safely mergeable via naive consensus into a single cube covering exact same minterms:
                # They must have exactly the same DCs everywhere except the single merge var.
                # Actually, a safer and strict definition:
                # diff_on & diff_off must both be exactly the same single bit position, 
                # AND at that bit, one cube is 1 and the other is 0.
                if diff_on == diff_off and diff_on != 0 and (diff_on & (diff_on - 1)) == 0:
                    # differ in exactly one bit
                    # Also must be completely identical everywhere else
                    # This means their masking out that bit makes them equal
                    mask_out = all_ones ^ diff_on
                    if (on_i & mask_out) == (on_j & mask_out) and (off_i & mask_out) == (off_j & mask_out):
                        # Merge them! The varying bit becomes dc (1 inside both on and off)
                        new_on = on_i | diff_on
                        new_off = off_i | diff_on
                        new_cubes.append((new_on, new_off))
                        keep[i] = False
                        keep[j] = False
                        merged_this = True
                        merged_any = True
                        break
            
            if not merged_this:
                new_cubes.append((on_i, off_i))

        cubes = new_cubes

    return cubes


def _remove_contained_cubes(cubes: List[Tuple[int, int]], all_ones: int) -> List[Tuple[int, int]]:
    n = len(cubes)
    if n <= 1: return cubes
    
    processed = []
    for on, off in cubes:
        care = (on ^ off) & all_ones
        processed.append((care.bit_count(), care, on, off))
        
    processed.sort(key=lambda x: x[0])
    
    keep = []
    for bc_i, care_i, on_i, off_i in processed:
        is_subsumed = False
        for care_j, on_j, off_j in keep:
            if (care_j & ~care_i) == 0 and (on_j & care_j) == (on_i & care_j):
                is_subsumed = True
                break
        if not is_subsumed:
            keep.append((care_i, on_i, off_i))
            
    return [(on, off) for care, on, off in keep]


# ======================================================================
# Complement via URP   (F_bar = x * Fx_bar  +  x' * Fx'_bar)
# ======================================================================

def _complement(
    cubes: List[Tuple[int, int]], 
    active_vars: int, 
    num_vars: int, 
    all_ones: int, 
    deadline: float = 0
) -> List[Tuple[int, int]]:
    """Recursively compute the complement of *cubes*."""

    if not cubes:
        return [(all_ones, all_ones)]

    for on, off in cubes:
        if on == all_ones and off == all_ones:
            return []

    # Filter subsumption (optional but helps keep cover clean before cofactoring)
    cubes = _remove_contained_cubes(cubes, all_ones)
    if not cubes:
        return [(all_ones, all_ones)]
    for on, off in cubes:
        if on == all_ones and off == all_ones:
            return []

    if len(cubes) == 1:
        # single cube -> De Morgan's
        on_i, off_i = cubes[0]
        res = []
        for v in range(num_vars):
            vbit = 1 << (num_vars - 1 - v)
            if not (active_vars & vbit): continue
            b1 = (on_i & vbit)
            b0 = (off_i & vbit)
            if b1 and b0: continue # dc
            # if 1, compl is 0. if 0, compl is 1.
            n_on = all_ones
            n_off = all_ones
            if b1 and not b0:
                n_on &= ~vbit
            elif b0 and not b1:
                n_off &= ~vbit
            res.append((n_on, n_off))
        return res

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

        if c0 > 0 and c1 > 0:
            is_unate = False
            score = c0 + c1
            bal = abs(c1 - c0)
            if score > best_score or (score == best_score and bal < best_balance):
                best_var = v; best_score = score; best_balance = bal

    if is_unate:
        # For unate, we can technically just use the same cofactor splitting!
        # The true URP unate algorithm cofactors on a variable of the unate cover. 
        # But we must pick a variable that actually HAS a literal (i.e. c1 > 0 or c0 > 0).
        best_var = -1
        for v in range(num_vars):
            vbit = 1 << (num_vars - 1 - v)
            if not (active_vars & vbit): continue
            c1 = sum(1 for on, off in cubes if (on & vbit) and not (off & vbit))
            c0 = sum(1 for on, off in cubes if (off & vbit) and not (on & vbit))
            if c1 > 0 or c0 > 0:
                best_var = v
                break
        
        if best_var == -1:
            # all active vars are DC -> tautology
            return []

    if deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    vbit = 1 << (num_vars - 1 - best_var)
    new_active = active_vars & ~vbit

    pos_cubes = list(set((on | vbit, off | vbit) for on, off in cubes if not ((off & vbit) and not (on & vbit))))
    neg_cubes = list(set((on | vbit, off | vbit) for on, off in cubes if not ((on & vbit) and not (off & vbit))))

    compl_pos = _complement(pos_cubes, new_active, num_vars, all_ones, deadline)
    compl_neg = _complement(neg_cubes, new_active, num_vars, all_ones, deadline)

    result = []
    # F_bar = x * compl_pos  +  x' * compl_neg
    for on_c, off_c in compl_pos:
        result.append((on_c | vbit, off_c & ~vbit))
    for on_c, off_c in compl_neg:
        result.append((on_c & ~vbit, off_c | vbit))

    result = _merge_adjacent(result, all_ones)
    result = _remove_contained_cubes(result, all_ones)
    return result


# ======================================================================
# Public API
# ======================================================================

def generate_complement(cover: Cover, timeout: float = COMPLGEN_TIMEOUT) -> Cover:
    """Compute and return the complement of *cover* as a new Cover."""
    num_vars = cover.num_inputs
    all_ones = (1 << num_vars) - 1
    mask_cubes = [_str_to_masks(c, num_vars) for c in cover.cubes]
    deadline = time.perf_counter() + timeout if timeout else 0
    
    try:
        compl_masks = _complement(mask_cubes, all_ones, num_vars, all_ones, deadline)
        timed_out = False
    except ComplementTimeout:
        compl_masks = []
        timed_out = True
        
    compl_cubes = [_masks_to_str(on, off, num_vars) for on, off in compl_masks]
    
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
        print("Usage: python Ha_Lenhart_complgen.py <cover_file> [<cover_file2> ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
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
