"""
espresso_parser.py
──────────────────
Shared ESPRESSO PLA cover parser and writer.
Encoding: '0'→1, '1'→2, '-'→3 stored as uint8 NumPy arrays.

Usage (library):    from espresso_parser import Cover, parse_cover, write_cover
Usage (standalone): python espresso_parser.py <file1> [<file2> ...]
"""

from __future__ import annotations
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Cover:
    """
    Represents an ESPRESSO-format cover (PLA table).

    Attributes
    ----------
    num_inputs : int
        Number of input variables (.i).
    num_outputs : int
        Number of output columns (.o).  Always 1 in this homework.
    input_labels : list[str]
        Variable names from .ilb (may be empty if .ilb is absent).
    output_labels : list[str]
        Output names from .ob (may be empty if .ob is absent).
    cubes : np.ndarray, shape (P, num_inputs), dtype uint8
        Each row is one product term. Values: 1='0', 2='1', 3='-'.
        Only cubes with output '1' are stored.
    """
    num_inputs: int = 0
    num_outputs: int = 0
    input_labels: List[str] = field(default_factory=list)
    output_labels: List[str] = field(default_factory=list)
    cubes: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.uint8))

    # Convenience ──────────────────────────────────────────────────────

    @property
    def num_cubes(self) -> int:
        return len(self.cubes)

    def __repr__(self) -> str:
        return (
            f"Cover(inputs={self.num_inputs}, outputs={self.num_outputs}, "
            f"cubes={self.num_cubes}, "
            f"ilb={self.input_labels}, ob={self.output_labels})"
        )


# ──────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────

def parse_cover(filepath: str) -> Cover:
    """
    Parse an ESPRESSO PLA file and return a Cover object.

    Raises
    ------
    ValueError
        On any format inconsistency (wrong cube width, mismatched .p
        count, etc.).
    """
    cover = Cover()
    declared_p: Optional[int] = None  # .p value, if present
    parsed_cubes = []

    with open(filepath, "r") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue

            # End marker
            if line == ".e":
                break

            # ── Header directives ─────────────────────────────────────
            if line.startswith(".i "):
                cover.num_inputs = int(line.split()[1])
                continue

            if line.startswith(".o "):
                cover.num_outputs = int(line.split()[1])
                continue

            if line.startswith(".p "):
                declared_p = int(line.split()[1])
                continue

            if line.startswith(".ilb"):
                cover.input_labels = line.split()[1:]
                continue

            if line.startswith(".ob"):
                cover.output_labels = line.split()[1:]
                continue

            # Skip any other dot-directive we don't recognise
            if line.startswith("."):
                continue

            # ── Product-term line ─────────────────────────────────────
            # Format: <input_pattern> <output_pattern>
            # e.g.  "11- 1" or "1-1 1"
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Bad product-term line (expected input + output): '{line}'"
                )

            input_pattern = parts[0]
            output_value = parts[1]

            # Validate width matches .i
            if cover.num_inputs and len(input_pattern) != cover.num_inputs:
                raise ValueError(
                    f"Cube width {len(input_pattern)} != declared .i "
                    f"{cover.num_inputs}: '{line}'"
                )

            # Validate characters
            if not all(ch in "01-" for ch in input_pattern):
                raise ValueError(
                    f"Invalid character in input pattern: '{input_pattern}'"
                )

            # We only store cubes with output '1'
            if output_value == "1":
                row = []
                for ch in input_pattern:
                    if ch == '-': row.append(3)
                    elif ch == '1': row.append(2)
                    else: row.append(1)
                parsed_cubes.append(row)

    if parsed_cubes:
        cover.cubes = np.array(parsed_cubes, dtype=np.uint8)
    elif cover.num_inputs:
        cover.cubes = np.empty((0, cover.num_inputs), dtype=np.uint8)

    # ── Post-parse validation ─────────────────────────────────────────
    if declared_p is not None and declared_p != cover.num_cubes:
        raise ValueError(
            f".p declared {declared_p} cubes but parsed {cover.num_cubes}"
        )

    if cover.input_labels and len(cover.input_labels) != cover.num_inputs:
        raise ValueError(
            f".ilb has {len(cover.input_labels)} labels but .i = "
            f"{cover.num_inputs}"
        )

    if cover.output_labels and len(cover.output_labels) != cover.num_outputs:
        raise ValueError(
            f".ob has {len(cover.output_labels)} labels but .o = "
            f"{cover.num_outputs}"
        )

    return cover


# ──────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────

def write_cover(cover: Cover, filepath: str) -> None:
    """Write a Cover object back to an ESPRESSO PLA file."""
    # Ensure parent directory exists
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(filepath, "w") as fh:
        fh.write(f".i {cover.num_inputs}\n")
        fh.write(f".o {cover.num_outputs}\n")
        fh.write(f".p {cover.num_cubes}\n")
        if cover.input_labels:
            fh.write(f".ilb {' '.join(cover.input_labels)}\n")
        if cover.output_labels:
            fh.write(f".ob {' '.join(cover.output_labels)}\n")
        
        for cube in cover.cubes:
            cube_str = "".join(['-' if v == 3 else '1' if v == 2 else '0' for v in cube])
            fh.write(f"{cube_str} 1\n")
        fh.write(".e\n")


# ----------------------------------------------------------------------
# Output path helper
# ----------------------------------------------------------------------

def get_output_path(input_filepath: str, suffix: str,
                    category: str = "Results") -> str:
    """
    Compute the output file path for a given input file.

    Parameters
    ----------
    input_filepath : str
        Path to the input cover file.
    suffix : str
        Suffix to append to the basename (e.g. '_off_cube', '_report.txt').
    category : str
        Sub-category folder label appended after the input folder name.
        Examples: 'Tautcheck-Results', 'Tautcheck-Reports',
                  'Complgen-Results', 'Complgen-Reports'.

    Rules:
        - If the input file is in the current directory (no parent folder),
          the output is written next to it:  <basename><suffix>
        - If the input file is inside a subfolder, the output goes into a
          new sibling folder named  <input_folder>-<category>/:
              Tautology-Checking-Tests/TC_T5  (category='Tautcheck-Results')
              -> Tautology-Checking-Tests-Tautcheck-Results/TC_T5<suffix>

    The output directory is created automatically if it does not exist.
    """
    input_dir = os.path.dirname(input_filepath)
    basename = os.path.basename(input_filepath)
    output_filename = basename + suffix

    if input_dir:
        results_dir = input_dir.rstrip("/\\") + "-" + category
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, output_filename)
    else:
        return output_filename


# ──────────────────────────────────────────────────────────────────────
# CLI – quick sanity check
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python espresso_parser.py <file1> [<file2> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"  Parsing: {path}")
        print(f"{'='*60}")
        c = parse_cover(path)
        print(c)
        print(f"  First 5 cubes: {c.cubes[:5]}")
        print(f"  Last  5 cubes: {c.cubes[-5:]}")
