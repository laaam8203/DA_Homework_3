"""
equivalence_checker.py
──────────────────────
Checks whether two covers (in two files) represent the same boolean function.

Method:
    A == B  iff:
        1. A . B_bar = 0   (nothing in A is outside B)
        2. B . A_bar = 0   (nothing in B is outside A)

    Equivalently: (A + B_bar) is a tautology AND (B + A_bar) is a tautology.
    Or: check that (A xor B) = 0.

    We use the complement generator and tautology checker:
        - Compute complement of B
        - Check if A union complement(B) covers everything that A covers
          but B doesn't.  More precisely: check if (A intersect B_bar)
          has any minterms.  If yes, A covers something B doesn't.
        - Repeat symmetrically.

Usage:
    python equivalence_checker.py <cover_file_A> <cover_file_B>

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
from typing import List

from espresso_parser import Cover, parse_cover
from Ha_Lenhart_tautcheck import check_tautology
from Ha_Lenhart_complgen import generate_complement


def _intersect_covers(f: Cover, g: Cover) -> Cover:
    """Compute intersection of two covers."""
    assert f.num_inputs == g.num_inputs
    result_cubes: List[str] = []
    for cf in f.cubes:
        for cg in g.cubes:
            inter = _and_cube(cf, cg)
            if inter is not None:
                result_cubes.append(inter)
    return Cover(
        num_inputs=f.num_inputs,
        num_outputs=f.num_outputs,
        input_labels=f.input_labels,
        output_labels=f.output_labels,
        cubes=result_cubes,
    )


def _union_covers(f: Cover, g: Cover) -> Cover:
    """Union of two covers (concatenate cube lists)."""
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


def check_equivalence(a: Cover, b: Cover) -> bool:
    """
    Check if A and B represent the same boolean function.
    Prints detailed results and returns True if equivalent.
    """
    assert a.num_inputs == b.num_inputs
    passed = True

    # Direction 1: Is there anything in A that's not in B?
    # Compute B_bar, then check if A . B_bar has any minterms
    print("\n  Direction 1: A subset of B?")
    print("    Computing complement of B...")
    b_bar = generate_complement(b)
    print(f"    B_bar has {b_bar.num_cubes} cubes")

    if b_bar.num_cubes == 0:
        # B is a tautology, so A is trivially a subset
        print("    B is a tautology -> A is trivially a subset of B")
    else:
        a_and_bbar = _intersect_covers(a, b_bar)
        if a_and_bbar.num_cubes == 0:
            print("    PASS - A intersect B_bar is empty.")
        else:
            # Check if the intersection actually covers anything
            # (it has cubes so it covers at least one minterm)
            print(f"    FAIL - A intersect B_bar has {a_and_bbar.num_cubes} cubes.")
            print(f"    Witness (in A but not B): {a_and_bbar.cubes[0]}")
            passed = False

    # Direction 2: Is there anything in B that's not in A?
    print("\n  Direction 2: B subset of A?")
    print("    Computing complement of A...")
    a_bar = generate_complement(a)
    print(f"    A_bar has {a_bar.num_cubes} cubes")

    if a_bar.num_cubes == 0:
        print("    A is a tautology -> B is trivially a subset of A")
    else:
        b_and_abar = _intersect_covers(b, a_bar)
        if b_and_abar.num_cubes == 0:
            print("    PASS - B intersect A_bar is empty.")
        else:
            print(f"    FAIL - B intersect A_bar has {b_and_abar.num_cubes} cubes.")
            print(f"    Witness (in B but not A): {b_and_abar.cubes[0]}")
            passed = False

    return passed


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python equivalence_checker.py <cover_A> <cover_B>")
        sys.exit(1)

    a_path, b_path = sys.argv[1], sys.argv[2]

    print(f"\n{'='*60}")
    print(f"  Equivalence Check")
    print(f"  A: {os.path.basename(a_path)}")
    print(f"  B: {os.path.basename(b_path)}")
    print(f"{'='*60}")

    a = parse_cover(a_path)
    b = parse_cover(b_path)

    print(f"  A: {a.num_inputs} vars, {a.num_cubes} cubes")
    print(f"  B: {b.num_inputs} vars, {b.num_cubes} cubes")

    result = check_equivalence(a, b)

    print(f"\n  {'='*40}")
    if result:
        print(f"  RESULT: A and B are EQUIVALENT")
    else:
        print(f"  RESULT: A and B are NOT equivalent")
    print(f"  {'='*40}")


if __name__ == "__main__":
    main()
