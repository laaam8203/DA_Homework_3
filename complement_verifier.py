"""
complement_verifier.py
──────────────────────
Verifies whether two covers (in two files) are complements of each other.

Method:
    G = complement(F)  iff:
        1. G intersect F = empty   (G . F = 0)  -- no overlap
        2. G union F = tautology   (G + F = 1)  -- covers everything

Both checks use the tautology checker.

Cube encoding (matches espresso_parser.py):
    '0' → 1  (uint8)
    '1' → 2  (uint8)
    '-' → 3  (uint8)

Usage:
    python complement_verifier.py <cover_file_F> <cover_file_G>

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os

import numpy as np

from espresso_parser import Cover, parse_cover
from Ha_Lenhart_tautcheck import check_tautology


# ──────────────────────────────────────────────────────────────────────
# Cube / Cover operations (all in uint8 numpy encoding)
# ──────────────────────────────────────────────────────────────────────

def _and_cubes_vectorized(cf: np.ndarray, cg_block: np.ndarray) -> np.ndarray:
    """
    Intersect a single cube cf (shape: (n,)) with every cube in cg_block
    (shape: (M, n)).

    Encoding rules for intersection:
        dc  & x   → x          (3 & anything → anything)
        x   & x   → x          (same literal → that literal)
        x   & ~x  → empty      (1&2 or 2&1 → conflict)

    We use the following vectorized formula:
        result[j] = max(cf[j], cg[j])   when they're not conflicting
    Conflict: one is 1 and the other is 2 (or vice-versa).

    Returns the sub-array of cg_block rows that have a non-empty
    intersection with cf, with the intersection values filled in.
    """
    n = len(cf)
    # Broadcast cf across all M rows of cg_block
    cf_b = np.broadcast_to(cf, cg_block.shape)          # (M, n)

    # Conflict: one is 1 and the other is 2
    conflict = ((cf_b == 1) & (cg_block == 2)) | ((cf_b == 2) & (cg_block == 1))
    has_conflict = conflict.any(axis=1)                  # (M,) bool

    # Intersection value: element-wise max (dc=3 is transparent; literals win)
    inter = np.maximum(cf_b, cg_block)                  # (M, n)

    # Zero out conflicting rows (we'll filter them below)
    valid_rows = ~has_conflict
    return inter[valid_rows]


def _intersect_covers(f: Cover, g: Cover) -> Cover:
    """
    Compute F ∩ G: the set of all non-empty pairwise cube intersections.
    Returns a Cover whose cubes is a numpy uint8 array.
    """
    assert f.num_inputs == g.num_inputs
    n = f.num_inputs

    if f.num_cubes == 0 or g.num_cubes == 0:
        return Cover(
            num_inputs=n,
            num_outputs=f.num_outputs,
            input_labels=f.input_labels,
            output_labels=f.output_labels,
            cubes=np.empty((0, n), dtype=np.uint8),
        )

    parts = []
    for cf in f.cubes:
        chunk = _and_cubes_vectorized(cf, g.cubes)
        if len(chunk) > 0:
            parts.append(chunk)

    if parts:
        result_cubes = np.vstack(parts).astype(np.uint8)
    else:
        result_cubes = np.empty((0, n), dtype=np.uint8)

    return Cover(
        num_inputs=n,
        num_outputs=f.num_outputs,
        input_labels=f.input_labels,
        output_labels=f.output_labels,
        cubes=result_cubes,
    )


def _union_covers(f: Cover, g: Cover) -> Cover:
    """
    Compute F ∪ G: simple concatenation of cube arrays.
    """
    assert f.num_inputs == g.num_inputs
    n = f.num_inputs

    if f.num_cubes == 0 and g.num_cubes == 0:
        combined = np.empty((0, n), dtype=np.uint8)
    elif f.num_cubes == 0:
        combined = g.cubes
    elif g.num_cubes == 0:
        combined = f.cubes
    else:
        combined = np.vstack([f.cubes, g.cubes]).astype(np.uint8)

    return Cover(
        num_inputs=n,
        num_outputs=f.num_outputs,
        input_labels=f.input_labels,
        output_labels=f.output_labels,
        cubes=combined,
    )


def _cube_to_str(cube: np.ndarray) -> str:
    """Convert a uint8 cube row back to a human-readable string."""
    return ''.join('-' if v == 3 else '1' if v == 2 else '0' for v in cube)


# ──────────────────────────────────────────────────────────────────────
# Verification logic
# ──────────────────────────────────────────────────────────────────────

def verify_complement(f: Cover, g: Cover) -> bool:
    """
    Verify if G is the complement of F.
    Prints results and returns True if they are complements.
    """
    n = f.num_inputs
    assert n == g.num_inputs, "Covers must have the same number of inputs"

    passed = True

    # ── Check 1: F ∩ G = ∅ ────────────────────────────────────────────
    print("\n  Check 1: F intersect G = empty?")
    fg_inter = _intersect_covers(f, g)
    if fg_inter.num_cubes == 0:
        print("    PASS - Intersection is empty (no cubes).")
    else:
        print(f"    FAIL - Intersection has {fg_inter.num_cubes} cubes.")
        print(f"    Witness (from intersection): {_cube_to_str(fg_inter.cubes[0])}")
        passed = False

    # ── Check 2: F ∪ G = 1 ────────────────────────────────────────────
    print("\n  Check 2: F union G = tautology?")
    fg_union = _union_covers(f, g)
    is_taut, witness, _ = check_tautology(fg_union)
    if is_taut:
        print("    PASS - Union is a tautology.")
    else:
        print(f"    FAIL - Union is NOT a tautology.")
        print(f"    Witness (uncovered minterm): {witness}")
        passed = False

    return passed


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python complement_verifier.py <cover_F> <cover_G>")
        sys.exit(1)

    f_path, g_path = sys.argv[1], sys.argv[2]

    print(f"\n{'='*60}")
    print(f"  Complement Verification")
    print(f"  F: {os.path.basename(f_path)}")
    print(f"  G: {os.path.basename(g_path)}")
    print(f"{'='*60}")

    f = parse_cover(f_path)
    g = parse_cover(g_path)

    print(f"  F: {f.num_cubes} cubes over {f.num_inputs} variables")
    print(f"  G: {g.num_cubes} cubes over {g.num_inputs} variables")

    result = verify_complement(f, g)

    print(f"\n  {'='*40}")
    if result:
        print(f"  RESULT: G IS the complement of F  ✓")
    else:
        print(f"  RESULT: G is NOT the complement of F  ✗")
    print(f"  {'='*40}\n")


if __name__ == "__main__":
    main()
