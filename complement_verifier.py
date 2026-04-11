"""
complement_verifier.py
──────────────────────
Verifies whether two covers (in two files) are complements of each other.

Method:
    G = complement(F)  iff:
        1. G intersect F = empty   (G . F = 0)  -- no overlap
        2. G union F = tautology   (G + F = 1)  -- covers everything

Both checks use the tautology checker.

Usage:
    python complement_verifier.py <cover_file_F> <cover_file_G>

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
from typing import List

from espresso_parser import Cover, parse_cover, write_cover
from Ha_Lenhart_tautcheck import check_tautology


def _intersect_covers(f: Cover, g: Cover) -> Cover:
    """
    Compute the intersection (AND) of two covers.
    F . G = union of all pairwise cube intersections.
    """
    assert f.num_inputs == g.num_inputs
    n = f.num_inputs
    result_cubes: List[str] = []

    for cf in f.cubes:
        for cg in g.cubes:
            inter = _and_cube(cf, cg)
            if inter is not None:
                result_cubes.append(inter)

    return Cover(
        num_inputs=n,
        num_outputs=f.num_outputs,
        input_labels=f.input_labels,
        output_labels=f.output_labels,
        cubes=result_cubes,
    )


def _union_covers(f: Cover, g: Cover) -> Cover:
    """
    Compute the union (OR) of two covers.
    Simply concatenates the cube lists.
    """
    assert f.num_inputs == g.num_inputs
    return Cover(
        num_inputs=f.num_inputs,
        num_outputs=f.num_outputs,
        input_labels=f.input_labels,
        output_labels=f.output_labels,
        cubes=f.cubes + g.cubes,
    )


def _and_cube(a: str, b: str):
    """Intersect two cubes. Returns None if empty."""
    result = []
    for ca, cb in zip(a, b):
        if ca == '-':
            result.append(cb)
        elif cb == '-':
            result.append(ca)
        elif ca == cb:
            result.append(ca)
        else:
            return None
    return ''.join(result)


def verify_complement(f: Cover, g: Cover) -> bool:
    """
    Verify if G is the complement of F.
    Prints results and returns True if they are complements.
    """
    n = f.num_inputs
    assert n == g.num_inputs, "Covers must have the same number of inputs"

    passed = True

    # Check 1: G . F = 0
    print("\n  Check 1: F intersect G = empty?")
    fg_inter = _intersect_covers(f, g)
    if fg_inter.num_cubes == 0:
        print("    PASS - Intersection is empty (no cubes).")
    else:
        print(f"    FAIL - Intersection has {fg_inter.num_cubes} cubes.")
        print(f"    Witness (from intersection): {fg_inter.cubes[0]}")
        passed = False

    # Check 2: G + F = 1
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

    result = verify_complement(f, g)

    print(f"\n  {'='*40}")
    if result:
        print(f"  RESULT: G IS the complement of F")
    else:
        print(f"  RESULT: G is NOT the complement of F")
    print(f"  {'='*40}")


if __name__ == "__main__":
    main()
