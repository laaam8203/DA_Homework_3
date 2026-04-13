"""
Ha_Lenhart_bench_verify.py
──────────────────────────
Benchmark Ha_Lenhart_complgen against TC_B1, TC_B2, and TC_B6, then
verify each non-empty result with complement_verifier.py.

Outputs:
  • Per-benchmark timing, peak memory, and output-cube count.
  • PASS / FAIL verdict from complement_verifier.verify_complement.
  • A summary table comparing against the baseline recorded in reports.

Usage:
    python Ha_Lenhart_bench_verify.py [--skip-verify]

Options:
    --skip-verify   Skip the (potentially slow) complement verification step.

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
import time
import tracemalloc

# Allow running from any cwd by inserting this file's directory on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from espresso_parser import parse_cover, write_cover, get_output_path
from Ha_Lenhart_complgen import generate_complement
from complement_verifier import verify_complement

# ── Benchmarks to run ────────────────────────────────────────────────
BENCHMARKS = [
    os.path.join(_HERE, "Tautology-Checking-Benchmarks", "TC_B1"),
    os.path.join(_HERE, "Tautology-Checking-Benchmarks", "TC_B2"),
    os.path.join(_HERE, "Tautology-Checking-Benchmarks", "TC_B6"),
]

# ── Baseline from last recorded reports (for regression comparison) ──
BASELINE = {
    "TC_B1": {"time_s": 6.369917, "cubes": 4907},
    "TC_B2": {"time_s": 2.693168, "cubes": 0},
    "TC_B6": {"time_s": 0.851985, "cubes": 0},
}


def _hr(label: str = "") -> None:
    if label:
        pad = (58 - len(label)) // 2
        print(f"\n{'='*pad} {label} {'='*(58 - pad - len(label))}")
    else:
        print("=" * 60)


def run_benchmarks(skip_verify: bool = False) -> None:
    results = []

    for bm_path in BENCHMARKS:
        bm_name = os.path.basename(bm_path)
        _hr(bm_name)

        if not os.path.exists(bm_path):
            print(f"  [SKIP] File not found: {bm_path}")
            results.append({"name": bm_name, "skipped": True})
            continue

        cover = parse_cover(bm_path)
        print(f"  Variables   : {cover.num_inputs}")
        print(f"  Input cubes : {cover.num_cubes}")

        tracemalloc.start()
        t0 = time.perf_counter()
        compl_cover, timed_out, iters = generate_complement(cover)
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  Output cubes: {len(compl_cover.cubes)}")
        print(f"  Time        : {elapsed:.4f} s")
        print(f"  Peak memory : {peak_mem / 1024:.1f} KB")
        print(f"  Iterations  : {iters}")

        if timed_out:
            print("  Status      : TIMEOUT")
            results.append({"name": bm_name, "time_s": elapsed,
                             "cubes": len(compl_cover.cubes),
                             "timeout": True, "verify": "N/A"})
            continue

        # Write output so the verifier can reference a path if needed
        out_path = get_output_path(bm_path, "_compl", "Complgen-Results")
        write_cover(compl_cover, out_path)
        print(f"  Written to  : {out_path}")

        # ── Verification step ─────────────────────────────────────────
        verify_result = "SKIPPED"
        if not skip_verify:
            if len(compl_cover.cubes) == 0 and cover.num_cubes > 0:
                # Empty complement means input is a tautology — verify
                # by checking that F is itself a tautology (union = F = 1).
                print("\n  [Verify] Output is empty (input appears to be a tautology).")
                print("  [Verify] Checking F ∪ G = F alone is a tautology...")
                from Ha_Lenhart_tautcheck import check_tautology
                is_taut, witness, _ = check_tautology(cover)
                if is_taut:
                    print("  [Verify] PASS — F is a tautology; complement = empty is correct.")
                    verify_result = "PASS (tautology confirmed)"
                else:
                    print(f"  [Verify] FAIL — F is NOT a tautology but complement is empty!")
                    print(f"           Witness: {witness}")
                    verify_result = "FAIL"
            else:
                print(f"\n  [Verify] Running complement_verifier on {bm_name}...")
                passed = verify_complement(cover, compl_cover)
                verify_result = "PASS" if passed else "FAIL"
                print(f"  [Verify] Result: {verify_result}")

        results.append({
            "name":    bm_name,
            "time_s":  elapsed,
            "mem_kb":  peak_mem / 1024,
            "cubes":   len(compl_cover.cubes),
            "timeout": False,
            "verify":  verify_result,
        })

    # ── Summary table ─────────────────────────────────────────────────
    _hr("SUMMARY")
    col = "{:<8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>12}  {}"
    print(col.format("Bench", "Time(s)", "Base(s)", "Speedup", "Cubes", "Memory(KB)", "Verify"))
    print("-" * 78)
    for r in results:
        if r.get("skipped"):
            print(f"  {r['name']:<8}  SKIPPED")
            continue
        base = BASELINE.get(r["name"], {})
        base_t = base.get("time_s", float("nan"))
        speedup = f"{base_t / r['time_s']:.2f}x" if r["time_s"] > 0 else "N/A"
        timeout_flag = " [TIMEOUT]" if r.get("timeout") else ""
        print(col.format(
            r["name"],
            f"{r['time_s']:.4f}",
            f"{base_t:.4f}",
            speedup,
            str(r["cubes"]),
            f"{r.get('mem_kb', 0):.1f}",
            r["verify"] + timeout_flag,
        ))
    _hr()


if __name__ == "__main__":
    skip = "--skip-verify" in sys.argv
    run_benchmarks(skip_verify=skip)
