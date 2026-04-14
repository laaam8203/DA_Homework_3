"""
report_summary.py
-----------------
Parses all report files in a given Reports folder, generates a
comparative summary table (printed to terminal and saved as text),
and produces instrumentation plots.

Usage:
    python report_summary.py <reports_folder> [<reports_folder2> ...]

Examples:
    python report_summary.py Tautology-Checking-Tests-Tautcheck-Reports
    python report_summary.py Tautology-Checking-Tests-Complgen-Reports
    python report_summary.py Tautology-Checking-Tests-Tautcheck-Reports Tautology-Checking-Tests-Complgen-Reports

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
import re
import glob
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ======================================================================
# Report parser
# ======================================================================

def parse_report(filepath: str) -> Optional[Dict]:
    """Parse a single report text file and return a dict of fields."""
    data = {}
    data["file"] = os.path.basename(filepath)

    with open(filepath, "r") as f:
        text = f.read()

    # Detect report type
    if "Tautology Check:" in text:
        data["type"] = "tautcheck"
        m = re.search(r"Tautology Check:\s+(.+)", text)
        if m:
            data["benchmark"] = m.group(1).strip()
    elif "Complement Generation:" in text:
        data["type"] = "complgen"
        m = re.search(r"Complement Generation:\s+(.+)", text)
        if m:
            data["benchmark"] = m.group(1).strip()
    else:
        return None

    # Common fields
    m = re.search(r"Variables\s*:\s*(\d+)", text)
    if m:
        data["variables"] = int(m.group(1))

    m = re.search(r"Cubes\s*:\s*(\d+)", text)
    if m:
        data["cubes"] = int(m.group(1))

    m = re.search(r"Input cubes\s*:\s*(\d+)", text)
    if m:
        data["cubes"] = int(m.group(1))

    m = re.search(r"Result\s*:\s*(.+)", text)
    if m:
        data["result"] = m.group(1).strip()

    m = re.search(r"Witness\s*:\s*([01-]+)", text)
    if m:
        data["witness"] = m.group(1).strip()

    m = re.search(r"Output cubes\s*:\s*(\d+)", text)
    if m:
        data["output_cubes"] = int(m.group(1))

    # Instrumentation
    m = re.search(r"Execution time\s*:\s*([\d.]+)\s*s", text)
    if m:
        data["exec_time"] = float(m.group(1))

    m = re.search(r"Peak memory\s*:\s*([\d.]+)\s*KB", text)
    if m:
        data["peak_memory_kb"] = float(m.group(1))

    m = re.search(r"Max recursion depth\s*:\s*(\d+)", text)
    if m:
        data["max_depth"] = int(m.group(1))

    m = re.search(r"Base-case tautology\s*:\s*(\d+)", text)
    if m:
        data["base_taut"] = int(m.group(1))

    m = re.search(r"Base-case NOT taut\s*:\s*(\d+)", text)
    if m:
        data["base_not_taut"] = int(m.group(1))

    m = re.search(r"Unate reductions\s*:\s*(\d+)", text)
    if m:
        data["unate_reductions"] = int(m.group(1))

    m = re.search(r"Binate splits\s*:\s*(\d+)", text)
    if m:
        data["binate_splits"] = int(m.group(1))

    return data


def parse_folder(folder: str) -> List[Dict]:
    """Parse all report files in a folder, sorted by benchmark name."""
    reports = []
    pattern = os.path.join(folder, "*.txt")
    for fpath in sorted(glob.glob(pattern)):
        data = parse_report(fpath)
        if data:
            reports.append(data)
    return reports


# ======================================================================
# Table formatting
# ======================================================================

def _sort_key(report):
    """Sort reports by benchmark name with natural number ordering."""
    name = report.get("benchmark", "")
    parts = re.split(r'(\d+)', name)
    result = []
    for p in parts:
        if p.isdigit():
            result.append(int(p))
        else:
            result.append(p)
    return result


def generate_tautcheck_table(reports: List[Dict]) -> str:
    """Generate a formatted table for tautology checker reports."""
    reports = sorted(reports, key=_sort_key)
    lines = []
    lines.append("=" * 120)
    lines.append("  TAUTOLOGY CHECKER - COMPARATIVE SUMMARY")
    lines.append("=" * 120)

    header = (f"  {'Benchmark':<12} {'Vars':>5} {'Cubes':>8} {'Result':<16} "
              f"{'Time (s)':>10} {'Mem (KB)':>10} {'Depth':>6} "
              f"{'BaseTaut':>9} {'BaseNot':>8} {'Unate':>6} {'Binate':>7}")
    lines.append(header)
    lines.append("  " + "-" * 116)

    for r in reports:
        name = r.get("benchmark", "?")
        vars_ = r.get("variables", 0)
        cubes = r.get("cubes", 0)
        result = r.get("result", "?")
        time_ = r.get("exec_time", 0)
        mem = r.get("peak_memory_kb", 0)
        depth = r.get("max_depth", 0)
        bt = r.get("base_taut", 0)
        bnt = r.get("base_not_taut", 0)
        unate = r.get("unate_reductions", 0)
        binate = r.get("binate_splits", 0)

        line = (f"  {name:<12} {vars_:>5} {cubes:>8} {result:<16} "
                f"{time_:>10.6f} {mem:>10.2f} {depth:>6} "
                f"{bt:>9} {bnt:>8} {unate:>6} {binate:>7}")
        lines.append(line)

    lines.append("=" * 120)
    return "\n".join(lines)


def generate_complgen_table(reports: List[Dict]) -> str:
    """Generate a formatted table for complement generator reports."""
    reports = sorted(reports, key=_sort_key)
    lines = []
    lines.append("=" * 90)
    lines.append("  COMPLEMENT GENERATOR - COMPARATIVE SUMMARY")
    lines.append("=" * 90)

    header = (f"  {'Benchmark':<12} {'Vars':>5} {'In Cubes':>9} {'Out Cubes':>10} "
              f"{'Time (s)':>10} {'Mem (KB)':>10}")
    lines.append(header)
    lines.append("  " + "-" * 86)

    for r in reports:
        name = r.get("benchmark", "?")
        vars_ = r.get("variables", 0)
        cubes = r.get("cubes", 0)
        out_cubes = r.get("output_cubes", 0)
        time_ = r.get("exec_time", 0)
        mem = r.get("peak_memory_kb", 0)

        line = (f"  {name:<12} {vars_:>5} {cubes:>9} {out_cubes:>10} "
                f"{time_:>10.6f} {mem:>10.2f}")
        lines.append(line)

    lines.append("=" * 90)
    return "\n".join(lines)


# ======================================================================
# Plot generation
# ======================================================================

def _apply_yscale(ax, values: list, ylabel: str, log_threshold: float = 50.0) -> None:
    """Switch to log scale when max/min exceeds log_threshold; annotate ylabel."""
    pos = [v for v in values if v > 0]
    if pos and max(pos) / min(pos) >= log_threshold:
        ax.set_yscale("log")
        ax.set_ylabel(f"{ylabel}  [log scale]")
    else:
        ax.set_ylabel(ylabel)


def _style_xaxis(ax, names: list) -> None:
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)


def generate_tautcheck_plots(reports: List[Dict], output_dir: str) -> List[str]:
    """Produce a single 2×2 combined figure for all four tautcheck metrics.

    Y-axes automatically switch to log scale when the value range across
    benchmarks exceeds 50×, so small benchmarks remain visible alongside
    large ones on the same panel.
    """
    if not HAS_MATPLOTLIB or not reports:
        return []

    reports = sorted(reports, key=_sort_key)
    names  = [r.get("benchmark", "?") for r in reports]
    n      = len(names)
    x      = list(range(n))

    times   = [r.get("exec_time",        0) for r in reports]
    mems    = [r.get("peak_memory_kb",   0) for r in reports]
    depths  = [r.get("max_depth",        0) for r in reports]
    unates  = [r.get("unate_reductions", 0) for r in reports]
    binates = [r.get("binate_splits",    0) for r in reports]
    c_bars  = ["#2ecc71" if r.get("result", "") == "TAUTOLOGY" else "#e74c3c"
               for r in reports]

    style = "seaborn-v0_8-darkgrid"
    if style in plt.style.available:
        plt.style.use(style)

    fig_w = max(14, n * 0.9)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10))
    fig.suptitle("Tautology Checker — Benchmark Summary", fontsize=13,
                 fontweight="bold")

    # ── Execution Time (top-left) ──────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x, times, color=c_bars, edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    _apply_yscale(ax, times, "Execution Time (s)")
    ax.set_title("Execution Time")
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#2ecc71", label="Tautology"),
        plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="Not Tautology"),
    ], fontsize=8, loc="upper left")

    # ── Peak Memory (top-right) ────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(x, mems, color="#3498db", edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    _apply_yscale(ax, mems, "Peak Memory (KB)")
    ax.set_title("Peak Memory Usage")

    # ── Recursion Depth (bottom-left) ─────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x, depths, color="#9b59b6", edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    _apply_yscale(ax, depths, "Max Recursion Depth")
    ax.set_title("Max Recursion Depth")

    # ── Binate Splits & Unate Reductions stacked (bottom-right) ───────
    ax = axes[1, 1]
    ax.bar(x, binates, color="#e67e22", edgecolor="white", linewidth=0.5,
           label="Binate Splits")
    ax.bar(x, unates, bottom=binates, color="#1abc9c", edgecolor="white",
           linewidth=0.5, label="Unate Reductions")
    _style_xaxis(ax, names)
    totals = [u + b for u, b in zip(unates, binates)]
    _apply_yscale(ax, totals, "Count")
    ax.set_title("Binate Splits & Unate Reductions")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "tautcheck_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def generate_complgen_plots(reports: List[Dict], output_dir: str) -> List[str]:
    """Produce a single 2×2 combined figure for all four complgen metrics.

    The fourth panel shows output-cube count as a fraction of input-cube
    count (complement density), which is more informative than absolute
    counts when input sizes span several orders of magnitude.

    Y-axes automatically switch to log scale when the value range exceeds 50×.
    """
    if not HAS_MATPLOTLIB or not reports:
        return []

    reports = sorted(reports, key=_sort_key)
    names    = [r.get("benchmark", "?") for r in reports]
    n        = len(names)
    x        = list(range(n))

    times     = [r.get("exec_time",      0) for r in reports]
    mems      = [r.get("peak_memory_kb", 0) for r in reports]
    in_cubes  = [r.get("cubes",          0) for r in reports]
    out_cubes = [r.get("output_cubes",   0) for r in reports]
    ratios    = [o / i * 100 if i > 0 else 0
                 for o, i in zip(out_cubes, in_cubes)]

    style = "seaborn-v0_8-darkgrid"
    if style in plt.style.available:
        plt.style.use(style)

    fig_w = max(14, n * 0.9)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10))
    fig.suptitle("Complement Generator — Benchmark Summary", fontsize=13,
                 fontweight="bold")

    # ── Execution Time (top-left) ──────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x, times, color="#e74c3c", edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    _apply_yscale(ax, times, "Execution Time (s)")
    ax.set_title("Execution Time")

    # ── Peak Memory (top-right) ────────────────────────────────────────
    ax = axes[0, 1]
    ax.bar(x, mems, color="#3498db", edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    _apply_yscale(ax, mems, "Peak Memory (KB)")
    ax.set_title("Peak Memory Usage")

    # ── Input vs Output Cubes grouped (bottom-left) ────────────────────
    ax = axes[1, 0]
    w = 0.35
    ax.bar([i - w / 2 for i in x], in_cubes,  w, color="#2ecc71",
           edgecolor="white", linewidth=0.5, label="Input Cubes")
    ax.bar([i + w / 2 for i in x], out_cubes, w, color="#e67e22",
           edgecolor="white", linewidth=0.5, label="Output Cubes")
    _style_xaxis(ax, names)
    _apply_yscale(ax, in_cubes + out_cubes, "Cube Count")
    ax.set_title("Input vs Output Cubes")
    ax.legend(fontsize=8)

    # ── Complement density: output / input % (bottom-right) ────────────
    ax = axes[1, 1]
    ax.bar(x, ratios, color="#8e44ad", edgecolor="white", linewidth=0.5)
    _style_xaxis(ax, names)
    ax.set_ylabel("Output / Input  (%)")
    ax.set_title("Complement Density  (output cubes / input cubes)")
    # Always linear — ratios are 0–100 %
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "complgen_combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python report_summary.py <reports_folder> [<reports_folder2> ...]")
        print()
        print("Examples:")
        print("  python report_summary.py Tautology-Checking-Tests-Tautcheck-Reports")
        print("  python report_summary.py Tautology-Checking-Tests-Complgen-Reports")
        sys.exit(1)

    for folder in sys.argv[1:]:
        if not os.path.isdir(folder):
            print(f"  ERROR: '{folder}' is not a directory. Skipping.")
            continue

        reports = parse_folder(folder)
        if not reports:
            print(f"  No report files found in '{folder}'. Skipping.")
            continue

        report_type = reports[0].get("type", "unknown")

        # Generate table
        if report_type == "tautcheck":
            table = generate_tautcheck_table(reports)
        elif report_type == "complgen":
            table = generate_complgen_table(reports)
        else:
            print(f"  Unknown report type in '{folder}'. Skipping.")
            continue

        # Print to terminal
        print()
        print(table)

        # Save summary text file
        summary_path = os.path.join(folder, "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(table + "\n")
        print(f"\n  Summary saved to: {summary_path}")

        # Generate plots
        if HAS_MATPLOTLIB:
            if report_type == "tautcheck":
                plots = generate_tautcheck_plots(reports, folder)
            else:
                plots = generate_complgen_plots(reports, folder)

            for p in plots:
                print(f"  Plot saved to: {p}")
        else:
            print("  (matplotlib not available - skipping plots)")


if __name__ == "__main__":
    main()
