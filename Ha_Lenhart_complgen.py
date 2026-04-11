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

from espresso_parser import Cover, parse_cover, write_cover


# ======================================================================
# Core helpers (shared with tautology checker logic)
# ======================================================================

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
    """A cover is unate if every variable appears in only one polarity."""
    for v in range(num_vars):
        c0, c1, _ = _column(cubes, v)
        if c0 > 0 and c1 > 0:
            return False
    return True


def _has_all_dc_row(cubes: List[str]) -> bool:
    """True if some cube is all don't-cares."""
    return any(all(ch == '-' for ch in c) for c in cubes)


def _cofactor(cubes: List[str], var: int, val: str) -> List[str]:
    """Cofactor the cube list with respect to variable *var* = *val*."""
    opp = '1' if val == '0' else '0'
    result = []
    for cube in cubes:
        ch = cube[var]
        if ch == opp:
            continue
        new_cube = cube[:var] + '-' + cube[var + 1:]
        result.append(new_cube)
    return result


def _pick_binate_variable(cubes: List[str], num_vars: int) -> int:
    """Choose the most binate variable for splitting."""
    best_var = -1
    best_literal_count = -1
    best_balance = float('inf')

    for v in range(num_vars):
        c0, c1, cd = _column(cubes, v)
        if c0 == 0 or c1 == 0:
            continue
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
    """
    Intersect two cubes.  Returns None if the intersection is empty.
    Each position:
        '-' & x  = x
         x  & x  = x
        '0' & '1' = empty
    """
    result = []
    for ca, cb in zip(a, b):
        if ca == '-':
            result.append(cb)
        elif cb == '-':
            result.append(ca)
        elif ca == cb:
            result.append(ca)
        else:
            return None  # empty intersection
    return ''.join(result)


def _and_cube_with_literal(cube: str, var: int, val: str) -> str | None:
    """
    Intersect *cube* with the literal  x_var = val.
    Returns None if the result is empty.
    """
    ch = cube[var]
    if ch == val or ch == '-':
        return cube[:var] + val + cube[var + 1:]
    return None  # conflict


# ======================================================================
# Complement via URP   (F_bar = x * Fx_bar  +  x' * Fx'_bar)
# ======================================================================

def _complement(cubes: List[str], num_vars: int) -> List[str]:
    """
    Recursively compute the complement of *cubes*.
    Returns a list of cubes representing the complement cover.
    """

    # -- Base case: empty cover -> complement is the whole space -------
    if not cubes:
        return ['-' * num_vars]

    # -- Base case: tautology -> complement is empty -------------------
    if _has_all_dc_row(cubes):
        return []

    # -- Base case: single cube ----------------------------------------
    if len(cubes) == 1:
        return _complement_single_cube(cubes[0], num_vars)

    # -- Unate cover ---------------------------------------------------
    if _is_unate(cubes, num_vars):
        return _complement_unate(cubes, num_vars)

    # -- Binate split (Shannon expansion) ------------------------------
    var = _pick_binate_variable(cubes, num_vars)

    pos_cubes = _cofactor(cubes, var, '1')
    neg_cubes = _cofactor(cubes, var, '0')

    compl_pos = _complement(pos_cubes, num_vars)
    compl_neg = _complement(neg_cubes, num_vars)

    # F_bar = x_var * compl_pos  +  x_var' * compl_neg
    result = []
    for c in compl_pos:
        new_c = c[:var] + '1' + c[var + 1:]
        result.append(new_c)
    for c in compl_neg:
        new_c = c[:var] + '0' + c[var + 1:]
        result.append(new_c)

    # Merge adjacent cubes where possible (simple SCC reduction)
    result = _merge_adjacent(result, num_vars)

    return result


def _complement_single_cube(cube: str, num_vars: int) -> List[str]:
    """
    Complement of a single cube via De Morgan's law.
    E.g.  complement of '1-0' is ['0--', '--1']
    Only positions that are '0' or '1' contribute a cube.
    """
    result = []
    for i, ch in enumerate(cube):
        if ch == '-':
            continue
        new_cube = ['-'] * num_vars
        new_cube[i] = '1' if ch == '0' else '0'
        result.append(''.join(new_cube))
    return result


def _complement_unate(cubes: List[str], num_vars: int) -> List[str]:
    """
    Complement of a unate cover.

    For a unate cover, the complement can be computed by complementing
    each variable's "active" column and combining.  We use a recursive
    approach: pick a variable that appears as a literal, cofactor, and
    recombine.
    """
    # Find a variable that has literals (not all dc)
    for v in range(num_vars):
        c0, c1, cd = _column(cubes, v)
        if c0 > 0 or c1 > 0:
            # This variable has at least some non-dc entries
            if c0 > 0 and c1 > 0:
                # Should not happen in unate cover
                break
            if c1 > 0:
                # Positive unate in this variable
                # F = x_v * F_xv  + F_dc  (where F_dc are cubes with '-' in col v)
                # F_bar = x_v' * (F_dc)_bar  +  x_v * (F_xv union F_dc)_bar
                #       = x_v' * (F_dc)_bar_restricted  + ...
                # Simpler: cofactor approach
                pos_cubes = _cofactor(cubes, v, '1')
                neg_cubes = _cofactor(cubes, v, '0')
                # neg_cubes are just the dc rows (since positive unate, no '0')
                compl_pos = _complement(pos_cubes, num_vars)
                compl_neg = _complement(neg_cubes, num_vars)
                result = []
                for c in compl_pos:
                    result.append(c[:v] + '1' + c[v + 1:])
                for c in compl_neg:
                    result.append(c[:v] + '0' + c[v + 1:])
                return _merge_adjacent(result, num_vars)
            else:
                # Negative unate
                pos_cubes = _cofactor(cubes, v, '1')
                neg_cubes = _cofactor(cubes, v, '0')
                compl_pos = _complement(pos_cubes, num_vars)
                compl_neg = _complement(neg_cubes, num_vars)
                result = []
                for c in compl_pos:
                    result.append(c[:v] + '1' + c[v + 1:])
                for c in compl_neg:
                    result.append(c[:v] + '0' + c[v + 1:])
                return _merge_adjacent(result, num_vars)

    # If all columns are dc, cover is tautology → complement is empty
    return []


# ======================================================================
# Simple SCC-minimality helper: merge adjacent cubes
# ======================================================================

def _merge_adjacent(cubes: List[str], num_vars: int) -> List[str]:
    """
    Repeatedly merge pairs of cubes that differ in exactly one variable
    (one has '0', the other '1', rest identical).
    This produces a more compact (SCC-minimal-like) result.
    """
    changed = True
    while changed:
        changed = False
        new_cubes = []
        used = [False] * len(cubes)
        for i in range(len(cubes)):
            if used[i]:
                continue
            merged = False
            for j in range(i + 1, len(cubes)):
                if used[j]:
                    continue
                m = _try_merge(cubes[i], cubes[j], num_vars)
                if m is not None:
                    new_cubes.append(m)
                    used[i] = True
                    used[j] = True
                    merged = True
                    changed = True
                    break
            if not merged:
                new_cubes.append(cubes[i])
        cubes = new_cubes

    # Remove cubes that are contained by another cube (superset check)
    cubes = _remove_contained(cubes)

    return cubes


def _try_merge(a: str, b: str, num_vars: int) -> str | None:
    """If a and b differ in exactly one position (0 vs 1), merge to '-'."""
    diff_count = 0
    diff_pos = -1
    for i in range(num_vars):
        if a[i] != b[i]:
            diff_count += 1
            diff_pos = i
            if diff_count > 1:
                return None
    if diff_count == 1 and a[diff_pos] in '01' and b[diff_pos] in '01':
        return a[:diff_pos] + '-' + a[diff_pos + 1:]
    return None


def _cube_contains(big: str, small: str) -> bool:
    """True if cube *big* is a superset of cube *small*."""
    for a, b in zip(big, small):
        if a == '-':
            continue
        if a != b:
            return False
    return True


def _remove_contained(cubes: List[str]) -> List[str]:
    """Remove cubes that are strictly contained by another cube."""
    keep = []
    for i, ci in enumerate(cubes):
        contained = False
        for j, cj in enumerate(cubes):
            if i == j:
                continue
            if _cube_contains(cj, ci) and not (ci == cj and j > i):
                contained = True
                break
        if not contained:
            keep.append(ci)
    return keep


# ======================================================================
# Public API
# ======================================================================

def generate_complement(cover: Cover) -> Cover:
    """Compute and return the complement of *cover* as a new Cover."""
    compl_cubes = _complement(cover.cubes, cover.num_inputs)
    return Cover(
        num_inputs=cover.num_inputs,
        num_outputs=cover.num_outputs,
        input_labels=cover.input_labels,
        output_labels=cover.output_labels,
        cubes=compl_cubes,
    )


# ======================================================================
# CLI entry point
# ======================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <cover_file> [<cover_file2> ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"  Complement Generation: {os.path.basename(filepath)}")
        print(f"{'='*60}")

        cover = parse_cover(filepath)
        print(f"  Variables    : {cover.num_inputs}")
        print(f"  Input cubes  : {cover.num_cubes}")

        tracemalloc.start()
        t_start = time.perf_counter()

        compl = generate_complement(cover)

        t_end = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        elapsed = t_end - t_start

        # Write complement file
        out_path = filepath + "_compl"
        write_cover(compl, out_path)

        print(f"  Output cubes : {compl.num_cubes}")
        print(f"  Written to   : {out_path}")

        # Stats
        print(f"\n  -- Instrumentation --")
        print(f"  Execution time : {elapsed:.6f} s")
        print(f"  Peak memory    : {peak_mem / 1024:.2f} KB")


if __name__ == "__main__":
    main()
