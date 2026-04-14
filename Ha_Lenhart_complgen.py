"""
Ha_Lenhart_complgen.py
Complement Generation for ESPRESSO PLA covers.

Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring and
the lecture-theorem optimisation for unate variables.

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
import re
import time
import tracemalloc
import numpy as np
from typing import List, Tuple

from espresso_parser import Cover, parse_cover, write_cover, get_output_path

COMPLGEN_TIMEOUT     = 60 * 60  # 1-hour wall-clock cap

# ── Tuning knobs ──────────────────────────────────────────────────────
MEMO_MAX_CUBES        = 15   # memoize _complement calls for covers this small or smaller


class ComplementTimeout(Exception):
    pass


# ======================================================================
# Bitvector helpers  (2-bit encoding per variable per cube)
# ======================================================================
#   '0' -> (m1=0, m0=1),  '1' -> (m1=1, m0=0),  '-' -> (m1=1, m0=1)
#
# Each cube is packed into one uint64 per word (m1 and m0), supporting
# up to 64 variables.  All current benchmarks (max 25 vars) fit easily.

def _pack_cubes(cubes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """uint8 cover (N x V) -> packed (m1, m0) arrays.

    Uses the minimum sufficient integer width for better memory efficiency
    and bandwidth: uint16 for ≤16 vars, uint32 for ≤32, uint64 otherwise.
    All current benchmarks (≤32 vars) therefore use uint32, halving the
    memory bandwidth consumed by every downstream bitvector operation.
    """
    n, v = cubes.shape
    dt = np.uint16 if v <= 16 else (np.uint32 if v <= 32 else np.uint64)
    m1 = np.zeros(n, dtype=dt)
    m0 = np.zeros(n, dtype=dt)
    one = dt(1)
    for j in range(v):
        col  = cubes[:, j].astype(dt)
        jbit = one << dt(j)
        m1  |= np.where((col >> one) & one, jbit, dt(0))
        m0  |= np.where( col         & one, jbit, dt(0))
    return m1, m0


def _unpack_cubes(m1: np.ndarray, m0: np.ndarray, num_vars: int) -> np.ndarray:
    """Packed (m1, m0) arrays -> uint8 cover (N x V)."""
    n  = len(m1)
    dt = m1.dtype
    cubes = np.empty((n, num_vars), dtype=np.uint8)
    one = dt.type(1)
    for j in range(num_vars):
        b1 = ((m1 >> dt.type(j)) & one).astype(np.uint8)
        b0 = ((m0 >> dt.type(j)) & one).astype(np.uint8)
        cubes[:, j] = (b1 << np.uint8(1)) | b0
    return cubes


# ======================================================================
# Hash-based deduplication  (replaces np.unique(arr, axis=0))
# ======================================================================

def _fast_unique(cubes: np.ndarray) -> np.ndarray:
    """Remove duplicate rows in O(N·V) using a hash set.

    Unlike np.unique this does NOT sort the output — it preserves the
    first occurrence order.  Dedup without lexsort is ~3-5x faster for
    the sizes encountered in mid-recursion results.
    """
    if len(cubes) <= 1:
        return cubes
    seen: set = set()
    keep: list = []
    for i in range(len(cubes)):
        key = cubes[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) == len(cubes):
        return cubes                            # no duplicates found
    return cubes[np.array(keep, dtype=np.intp)]


# ======================================================================
# Post-processing: SCC-minimal cover reduction  (bitvector-accelerated)
# ======================================================================

def _remove_contained_bv(m1: np.ndarray, m0: np.ndarray,
                          deadline: float) -> Tuple[np.ndarray, np.ndarray]:
    """Drop cubes subsumed by a larger (more general) cube.
    A ⊆ B  iff  (A_m1 | B_m1)==B_m1  and  (A_m0 | B_m0)==B_m0.

    Pre-allocates four work arrays outside the hot loop to eliminate
    per-iteration numpy heap allocation (~5 µs × N allocations saved).
    A fast m1-only pre-check short-circuits the m0 test for the common
    case where no cube contains cube i under the m1 dimension alone.
    """
    n = len(m1)
    if n <= 1:
        return m1, m0
    keep = np.ones(n, dtype=bool)
    dt   = m1.dtype
    # Pre-allocate: zero dynamic allocation inside the loop
    or1  = np.empty(n, dtype=dt)
    or0  = np.empty(n, dtype=dt)
    c1   = np.empty(n, dtype=bool)
    c0   = np.empty(n, dtype=bool)
    for i in range(n):
        if (i & 2047) == 0 and deadline and time.perf_counter() > deadline:
            raise ComplementTimeout()
        if not keep[i]:
            continue
        # m1-only check: necessary condition for containment — skip m0 if it fails
        np.bitwise_or(m1[i], m1, out=or1)
        np.equal(or1, m1, out=c1)
        c1[i] = False
        if not np.any(c1):
            continue
        # Full check: m0 must also satisfy containment
        np.bitwise_or(m0[i], m0, out=or0)
        np.equal(or0, m0, out=c0)
        np.logical_and(c1, c0, out=c1)
        if np.any(c1):
            keep[i] = False
    return m1[keep], m0[keep]


def _merge_adjacent_hash(m1: np.ndarray, m0: np.ndarray,
                          deadline: float) -> Tuple[np.ndarray, np.ndarray]:
    """Distance-1 cube merging, vectorised per variable position.

    Replaces the pure-Python dict loop with numpy lexsort per variable,
    giving ~10× speedup for large covers (e.g. 100k cubes × 32 vars).

    Algorithm per pass:
      For each bit position j (0 .. highest used bit):
        1. Select cubes where position j is non-DC and not yet used.
        2. Compute the merge key (m1|j_bit, m0|j_bit) for each candidate.
        3. lexsort candidates by key; scan for consecutive equal-key pairs.
        4. Merge each valid pair (both un-used) → emit one merged cube.
      Carry all un-merged cubes forward, then repeat until no merges.
      Finally run containment removal.

    Correctness: only cubes at the SAME position j can form a key match,
    so no cross-position false-positive collisions are possible.
    Complexity: O(V · N log N) per pass (vs. O(N·V) Python dict ops).
    """
    if len(m1) <= 1:
        return m1, m0

    dt = m1.dtype
    # Determine highest bit position actually used across all cubes
    all_bits = int(np.bitwise_or.reduce(m1) | np.bitwise_or.reduce(m0))
    num_bits = max(all_bits.bit_length(), 1)

    changed = True
    while changed:
        if deadline and time.perf_counter() > deadline:
            raise ComplementTimeout()
        changed = False
        n = len(m1)
        used     = np.zeros(n, dtype=bool)
        merged_m1: list = []
        merged_m0: list = []

        non_dc_all = m1 ^ m0   # bit j set ↔ position j is non-DC in that cube

        for j in range(num_bits):
            j_bit = dt.type(1) << dt.type(j)   # scalar of the array's own dtype

            # Candidates: non-DC at j, not yet consumed this pass
            mask = ((non_dc_all >> dt.type(j)) & dt.type(1)).astype(bool) & ~used
            candidates = np.where(mask)[0]
            if len(candidates) < 2:
                continue

            # Merge key: forcing position j to DC gives the target cube.
            # Cast to int64 for stable lexsort (safe: uint16/32/64 ≤ 2^63).
            km1 = (m1[candidates] | j_bit).astype(np.int64)
            km0 = (m0[candidates] | j_bit).astype(np.int64)

            # Sort candidates by (km1, km0); adjacent equal pairs → mergeable
            order  = np.lexsort((km0, km1))
            s_idx  = candidates[order]
            s_km1  = km1[order]
            s_km0  = km0[order]

            eq          = (s_km1[:-1] == s_km1[1:]) & (s_km0[:-1] == s_km0[1:])
            pair_starts = np.where(eq)[0]

            for ps in pair_starts:
                ia = s_idx[ps]
                ib = s_idx[ps + 1]
                if not used[ia] and not used[ib]:
                    merged_m1.append(m1[ia] | j_bit)
                    merged_m0.append(m0[ia] | j_bit)
                    used[ia] = True
                    used[ib] = True
                    changed = True

        if changed:
            keep_idx  = np.where(~used)[0]
            nk, nm    = len(keep_idx), len(merged_m1)
            m1_new    = np.empty(nk + nm, dtype=dt)
            m0_new    = np.empty(nk + nm, dtype=dt)
            m1_new[:nk] = m1[keep_idx]
            m0_new[:nk] = m0[keep_idx]
            if nm:
                m1_new[nk:] = np.array(merged_m1, dtype=dt)
                m0_new[nk:] = np.array(merged_m0, dtype=dt)
            m1, m0 = m1_new, m0_new

    return _remove_contained_bv(m1, m0, deadline)


# ======================================================================
# Local mid-recursion containment filter
# ======================================================================

def _local_filter(cubes: np.ndarray, deadline: float) -> np.ndarray:
    """Remove dominated cubes via bitvectors, then unpack back to uint8.

    Called when an intermediate result exceeds the adaptive filter_threshold
    (max(300, num_cubes // 20)) to prevent the cube count from compounding
    multiplicatively across recursion levels.  N scales with input cover size,
    so cost scales accordingly; the threshold keeps N proportional to the
    problem rather than using a single hard-coded value.
    """
    if len(cubes) <= 1:
        return cubes
    m1, m0 = _pack_cubes(cubes)
    m1, m0 = _remove_contained_bv(m1, m0, deadline)
    return _unpack_cubes(m1, m0, cubes.shape[1])


# ======================================================================
# Tautology bounding  (lightweight recursive check with depth cap)
# ======================================================================

def _is_tautology(cubes: np.ndarray, num_vars: int,
                  deadline: float,
                  max_depth: int = 100, _depth: int = 0) -> bool:
    """Return True if *cubes* covers the entire Boolean space.

    max_depth is a safety cap; callers should pass num_vars*2+5 so that
    genuine tautologies (which need up to V levels to confirm) always
    succeed.  For non-tautologies the early-exit checks (column gap,
    all-unate) prevent deep recursion, so the cap is rarely reached.
    """
    if _depth >= max_depth:
        return False                            # budget exhausted; cannot confirm
    if deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    # ── Base cases ────────────────────────────────────────────────────
    if len(cubes) == 0:
        return False
    if np.any(np.all(cubes == 3, axis=1)):
        return True

    c0 = np.sum(cubes == 1, axis=0)
    c1 = np.sum(cubes == 2, axis=0)
    cd = np.sum(cubes == 3, axis=0)

    # Column-gap: a variable missing one polarity and no DC → half-space uncoverable
    if np.any((c1 == 0) & (cd == 0)) or np.any((c0 == 0) & (cd == 0)):
        return False

    has_0 = c0 > 0
    has_1 = c1 > 0

    # All-unate → not a tautology
    if not np.any(has_0 & has_1):
        return False

    # Multi-variable unate reduction: collapse ALL unate variables at once
    strictly_unate = ~(has_0 & has_1) & (has_0 | has_1)
    if np.any(strictly_unate):
        indices = np.where(strictly_unate)[0]
        mask = np.ones(len(cubes), dtype=bool)
        for v in indices:
            mask &= (cubes[:, v] != (2 if has_1[v] else 1))
        new_cubes = cubes[mask].copy()
        new_cubes[:, indices] = 3
        if len(new_cubes) > 1000:
            new_cubes = _fast_unique(new_cubes)
        return _is_tautology(new_cubes, num_vars, deadline, max_depth, _depth + 1)

    # Binate split with short-circuit
    binate_mask = has_0 & has_1
    lit_counts  = c0 + c1
    balances    = np.abs(c1 - c0)
    valid_lit   = np.where(binate_mask, lit_counts, -1)
    candidates  = (valid_lit == np.max(valid_lit))
    var = int(np.argmin(np.where(candidates, balances, np.iinfo(np.int64).max)))

    pos = cubes[cubes[:, var] != 1].copy(); pos[:, var] = 3
    if not _is_tautology(pos, num_vars, deadline, max_depth, _depth + 1):
        return False                            # short-circuit

    neg = cubes[cubes[:, var] != 2].copy(); neg[:, var] = 3
    return _is_tautology(neg, num_vars, deadline, max_depth, _depth + 1)


# ======================================================================
# Core URP complement recursion
# ======================================================================

def _complement_single_cube(cube: np.ndarray, num_vars: int) -> np.ndarray:
    """De Morgan base case: complement of one cube.
    Each non-dc literal becomes its own cube with polarity flipped."""
    lits = np.where(cube != 3)[0]
    k = len(lits)
    if k == 0:                              # universal cube -> empty complement
        return np.empty((0, num_vars), dtype=np.uint8)
    res = np.full((k, num_vars), 3, dtype=np.uint8)
    res[np.arange(k), lits] = 3 - cube[lits]
    return res


def _complement(cubes: np.ndarray, num_vars: int,
                deadline: float, iterations: List[int], depth: int,
                memo: dict, filter_threshold: int) -> np.ndarray:
    """Recursive URP complement.  Returns the complement cover as uint8 array.

    Parameters
    ----------
    cubes      : current on-set cover (uint8, N x V)
    num_vars   : total variable count
    deadline   : perf_counter() value after which we raise ComplementTimeout
    iterations : single-element list used as a mutable counter
    depth      : current recursion depth (for local-filter decisions)
    memo       : memoisation cache shared across the entire top-level call;
                 maps frozenset-of-tobytes → copy of the result array.
                 Callers may mutate the returned array in-place, so the
                 cache always stores a separate copy.
    filter_threshold : dynamic threshold for triggering _local_filter.
    """
    iterations[0] += 1
    if (iterations[0] & 2047) == 0 and deadline and time.perf_counter() > deadline:
        raise ComplementTimeout()

    # ── Base cases ────────────────────────────────────────────────────
    if len(cubes) == 0:
        return np.full((1, num_vars), 3, dtype=np.uint8)   # empty -> universe
    if np.any(np.all(cubes == 3, axis=1)):
        return np.empty((0, num_vars), dtype=np.uint8)     # tautology -> empty
    if len(cubes) == 1:
        return _complement_single_cube(cubes[0], num_vars)

    # ── Memoisation lookup ────────────────────────────────────────────
    # Only cache small covers — large covers rarely repeat and hashing is
    # expensive.  We cache the result AFTER all transformations so the key
    # maps exactly to the final complement of this exact sub-cover.
    cache_key = None
    if len(cubes) <= MEMO_MAX_CUBES:
        cache_key = frozenset(row.tobytes() for row in cubes)
        if cache_key in memo:
            cached = memo[cache_key]
            # Always return a copy so the caller can mutate freely
            return cached.copy() if len(cached) > 0 else np.empty((0, num_vars), dtype=np.uint8)

    # ── Common-cube extraction ────────────────────────────────────────
    # F = C · F_rest  =>  F̄ = C̄  +  C · complement(F_rest)
    common = np.bitwise_or.reduce(cubes, axis=0)
    if np.any(common != 3):
        compl_common = _complement_single_cube(common, num_vars)
        rest_cubes   = cubes | (common ^ 3)             # strip common literals
        rest_cubes   = _fast_unique(rest_cubes)          # dedup after stripping
        compl_rest   = _complement(rest_cubes, num_vars, deadline, iterations, depth + 1, memo, filter_threshold)
        if len(compl_rest) > 0:
            compl_rest[:, common != 3] = common[common != 3]
            result = np.vstack([compl_common, compl_rest])
        else:
            result = compl_common
        if cache_key is not None:
            memo[cache_key] = result.copy()
        return result

    # ── Column statistics ─────────────────────────────────────────────
    c0 = np.sum(cubes == 1, axis=0)
    c1 = np.sum(cubes == 2, axis=0)
    has_0 = c0 > 0
    has_1 = c1 > 0

    # Cheap tautology shortcuts reusing c0/c1 already computed:
    cd = np.sum(cubes == 3, axis=0)
    has_gap    = np.any((c1 == 0) & (cd == 0)) or np.any((c0 == 0) & (cd == 0))
    has_binate = np.any(has_0 & has_1)

    # Only invoke the recursive tautology check when cheap shortcuts fail
    # AND the cover is large enough to make the check worthwhile.
    # Depth cap = num_vars*2+5 ensures genuine tautologies are confirmed;
    # non-tautologies exit early via column-gap / all-unate shortcuts.
    if not has_gap and has_binate and len(cubes) >= num_vars * num_vars:
        if _is_tautology(cubes, num_vars, deadline,
                         max_depth=num_vars * 2 + 5):
            result = np.empty((0, num_vars), dtype=np.uint8)
            if cache_key is not None:
                memo[cache_key] = result
            return result

    # ── Iterative multi-variable unate optimisation (lecture theorem) ─
    # For pos-unate x_j:
    #   F̄ = complement(F|x_j=1)        [large branch, x_j stays dc in result]
    #       + x̄_j · complement(F|x_j=0) [small branch, pin x̄_j on result]
    # Processing ALL unate variables in one sweep avoids redundant
    # re-analysis of the shrinking cover after each single-var cofactor.
    unate_mask = (~(has_0 & has_1)) & (has_0 | has_1)
    if np.any(unate_mask):
        u_idx = np.where(unate_mask)[0]
        u_idx = u_idx[np.argsort(-(c0 + c1)[u_idx])]   # most literals first

        res_list: list = []
        cur = cubes.copy()

        for vi in u_idx:
            vi  = int(vi)
            pos = bool(has_1[vi])
            lit = np.uint8(2 if pos else 1)             # literal value to exclude
            pin = np.uint8(1 if pos else 2)             # polarity to pin in result

            f_small = cur[cur[:, vi] != lit].copy()
            f_small[:, vi] = 3
            cs = _complement(f_small, num_vars, deadline, iterations, depth + 1, memo, filter_threshold)
            if len(cs) > 0:
                cs[:, vi] = pin
                res_list.append(cs)

            cur[:, vi] = 3                              # advance large chain

        # Single recursion for the fully-dc'd cover
        cl = _complement(cur, num_vars, deadline, iterations, depth + 1, memo, filter_threshold)
        if len(cl) > 0:
            res_list.append(cl)

        if not res_list:
            result = np.empty((0, num_vars), dtype=np.uint8)
        else:
            res = np.vstack(res_list)
            res = _fast_unique(res)
            if len(res) > filter_threshold:        # contain explosion early
                res = _local_filter(res, deadline)
            result = res

        if cache_key is not None:
            memo[cache_key] = result.copy()
        return result

    # ── Binate Shannon split ──────────────────────────────────────────
    # Variable selection: most literals (primary), most balanced (secondary)
    binate_mask = has_0 & has_1
    lit_counts  = c0 + c1
    balances    = np.abs(c1 - c0)
    valid_lit   = np.where(binate_mask, lit_counts, -1)
    candidates  = (valid_lit == np.max(valid_lit))
    var = int(np.argmin(np.where(candidates, balances, np.iinfo(np.int64).max)))

    pos_cubes = cubes[cubes[:, var] != 1].copy(); pos_cubes[:, var] = 3
    neg_cubes = cubes[cubes[:, var] != 2].copy(); neg_cubes[:, var] = 3

    cp = _complement(pos_cubes, num_vars, deadline, iterations, depth + 1, memo, filter_threshold)
    cn = _complement(neg_cubes, num_vars, deadline, iterations, depth + 1, memo, filter_threshold)

    res_list2: list = []
    if len(cp) > 0: cp[:, var] = 2; res_list2.append(cp)
    if len(cn) > 0: cn[:, var] = 1; res_list2.append(cn)

    if not res_list2:
        result = np.empty((0, num_vars), dtype=np.uint8)
    else:
        res = np.vstack(res_list2)
        res = _fast_unique(res)
        if len(res) > filter_threshold:
            res = _local_filter(res, deadline)
        result = res

    if cache_key is not None:
        memo[cache_key] = result.copy()
    return result


# ======================================================================
# Public API
# ======================================================================

def generate_complement(cover: Cover,
                        timeout: float = COMPLGEN_TIMEOUT) -> Cover:
    """Compute the exact complement of *cover* via URP.
    Returns (complement_cover, timed_out, iteration_count)."""
    deadline   = time.perf_counter() + timeout if timeout else 0
    iterations = [0]
    memo: dict = {}                             # per-call memoisation cache
    filter_threshold = max(300, cover.num_cubes // 20)
    try:
        compl = _complement(cover.cubes, cover.num_inputs, deadline, iterations, 0, memo, filter_threshold)
        if len(compl) > 1:
            compl = _fast_unique(compl)
            m1, m0 = _pack_cubes(compl)
            m1, m0 = _merge_adjacent_hash(m1, m0, deadline)
            compl  = _unpack_cubes(m1, m0, cover.num_inputs)
        timed_out = False
    except ComplementTimeout:
        compl     = np.empty((0, cover.num_inputs), dtype=np.uint8)
        timed_out = True

    return Cover(
        num_inputs=cover.num_inputs,
        num_outputs=cover.num_outputs,
        input_labels=cover.input_labels,
        output_labels=cover.output_labels,
        cubes=compl,
    ), timed_out, iterations[0]


# ======================================================================
# CLI driver
# ======================================================================

def _natural_key(s: str):
    """Sort key that orders embedded integers numerically.
    e.g. TC_B2 < TC_B10 instead of TC_B10 < TC_B2 (lexicographic)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <path_to_file_or_folder>")
        sys.exit(1)

    targets = []
    for arg_path in sys.argv[1:]:
        if os.path.exists(arg_path):
            if os.path.isdir(arg_path):
                for f in sorted(os.listdir(arg_path), key=_natural_key):
                    fp = os.path.join(arg_path, f)
                    if os.path.isfile(fp) and not f.startswith("."):
                        targets.append(fp)
            else:
                targets.append(arg_path)

    for filepath in targets:
        report = []
        report.append(f"{'='*60}")
        report.append(f"  Complement Generation: {os.path.basename(filepath)}")
        report.append(f"{'='*60}")

        cover = parse_cover(filepath)
        report.append(f"  Variables    : {cover.num_inputs}")
        report.append(f"  Input cubes  : {cover.num_cubes}")

        tracemalloc.start()
        t0 = time.perf_counter()
        compl_cover, timed_out, iters = generate_complement(cover)
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if timed_out:
            report.append(f"  Result       : TIMEOUT (exceeded {COMPLGEN_TIMEOUT // 60} min)")
            report.append(f"  Output cubes : {len(compl_cover.cubes)} (Partial/Invalid)")
        else:
            report.append(f"  Output cubes : {len(compl_cover.cubes)}")
            out_path = get_output_path(filepath, "_compl", "Complgen-Results")
            write_cover(compl_cover, out_path)
            report.append(f"  Written to   : {out_path}")

        report.append(f"")
        report.append(f"  -- Instrumentation --")
        report.append(f"  Execution time : {elapsed:.6f} s")
        report.append(f"  Peak memory    : {peak_mem / 1024:.2f} KB")
        report.append(f"")

        for line in report:
            print(line)

        report_path = get_output_path(filepath, "_complgen_report.txt", "Complgen-Reports")
        with open(report_path, "w") as f:
            f.write("\n".join(report) + "\n")
        print(f"  Report written to: {report_path}")


if __name__ == "__main__":
    main()
