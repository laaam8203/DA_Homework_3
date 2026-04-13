"""
Ha_Lenhart_complgen.py
Complement Generation for ESPRESSO PLA covers.

Uses the Unate Recursive Paradigm (URP) with Shannon cofactoring and
the lecture-theorem optimisation for unate variables.

Encoding: '0' -> 1, '1' -> 2, '-' -> 3  (uint8 arrays, internal).
Post-processing uses a packed 2-bit bitvector representation (uint64)
for O(1)-per-pair containment and distance-1 checks.

Optimisations applied:
  - Common-cube extraction before recursion
  - Multi-variable unate sweep in a single pass (lecture theorem)
  - Tautology bounding with depth cap scaled to num_vars*2+5
  - Hash-based O(N·V) distance-1 merging (replaces O(N²) pairwise scan)
  - Hash-based O(N·V) deduplication (replaces O(N·V·log N) lexsort)
  - Memoisation of small sub-cover complement results
  - Local containment filtering mid-recursion to prevent cube-count explosion

Authors: Ha, Lenhart
Course : VLSI Design Automation (EECE 5186C/6086C) - HW3
"""

from __future__ import annotations
import sys
import os
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
    """uint8 cover (N x V) -> packed (m1, m0) uint64 arrays."""
    n, v = cubes.shape
    m1 = np.zeros(n, dtype=np.uint64)
    m0 = np.zeros(n, dtype=np.uint64)
    for j in range(v):
        col = cubes[:, j].astype(np.uint64)
        m1 |= ((col >> np.uint64(1)) & np.uint64(1)) << np.uint64(j)
        m0 |= ( col                  & np.uint64(1)) << np.uint64(j)
    return m1, m0


def _unpack_cubes(m1: np.ndarray, m0: np.ndarray, num_vars: int) -> np.ndarray:
    """Packed (m1, m0) uint64 arrays -> uint8 cover (N x V)."""
    n = len(m1)
    cubes = np.empty((n, num_vars), dtype=np.uint8)
    for j in range(num_vars):
        b1 = ((m1 >> np.uint64(j)) & np.uint64(1)).astype(np.uint8)
        b0 = ((m0 >> np.uint64(j)) & np.uint64(1)).astype(np.uint8)
        cubes[:, j] = (b1 << 1) | b0
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
    A ⊆ B  iff  (A_m1 | B_m1)==B_m1  and  (A_m0 | B_m0)==B_m0."""
    n = len(m1)
    if n <= 1:
        return m1, m0
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if (i & 2047) == 0 and deadline and time.perf_counter() > deadline:
            raise ComplementTimeout()
        if not keep[i]:
            continue
        cont = ((m1[i] | m1) == m1) & ((m0[i] | m0) == m0)
        cont[i] = False
        if np.any(cont):
            keep[i] = False
    return m1[keep], m0[keep]


def _merge_adjacent_hash(m1: np.ndarray, m0: np.ndarray,
                          deadline: float) -> Tuple[np.ndarray, np.ndarray]:
    """O(N·V) hash-based distance-1 cube merging.

    Key insight: two cubes c_a and c_b are distance-1 at position j iff,
    when both have position j forced to DC, they become identical:
      (m1[a] | bit_j, m0[a] | bit_j) == (m1[b] | bit_j, m0[b] | bit_j)

    IMPORTANT: we include bit_j itself as the first element of the key so
    that two cubes can only match if they were both processed at the SAME
    position.  Without this, a cube with '0' at position p and DC at q
    could collide with a cube having DC at p and '1' at q even though
    they are distance-2, producing a spurious over-general merged cube.

    Algorithm per pass:
      1. For each cube i and each non-DC position j, emit key =
         (j_bit, m1[i]|j_bit, m0[i]|j_bit) into a position-keyed table.
      2. For every key with ≥2 cube entries, pick the first two un-used
         cubes and emit the merged cube (key[1], key[2]).
      3. Carry all un-merged cubes forward unchanged.
      4. Repeat until no merges occur, then run containment removal.

    Complexity: O(N·V) per pass (vs. O(N²) for pairwise comparison).
    """
    if len(m1) <= 1:
        return m1, m0

    changed = True
    while changed:
        if deadline and time.perf_counter() > deadline:
            raise ComplementTimeout()
        changed = False
        n = len(m1)

        # Build merge table: (j_bit, merged_m1, merged_m0) -> list of cube indices.
        # Keying on j_bit prevents cross-position false-positive collisions.
        key_to_cubes: dict = {}
        for i in range(n):
            # Non-DC positions: bit j is non-DC when m1[i] XOR m0[i] has bit j set.
            # (DC: both bits 1 → XOR = 0;  '0': m1=0,m0=1;  '1': m1=1,m0=0 → XOR = 1)
            non_dc = int(m1[i] ^ m0[i])
            temp = non_dc
            while temp:
                j_bit = temp & (-temp)          # isolate lowest set bit
                temp  &= temp - 1               # clear lowest set bit
                key = (j_bit, int(m1[i]) | j_bit, int(m0[i]) | j_bit)
                if key not in key_to_cubes:
                    key_to_cubes[key] = []
                key_to_cubes[key].append(i)

        used = [False] * n
        new_m1: list = []
        new_m0: list = []

        # Find mergeable pairs and merge them
        for key, indices in key_to_cubes.items():
            if len(indices) >= 2:
                a = b = -1
                for idx in indices:
                    if not used[idx]:
                        if a < 0:
                            a = idx
                        else:
                            b = idx
                            break
                if a >= 0 and b >= 0:
                    new_m1.append(np.uint64(key[1]))   # key[1] = merged m1
                    new_m0.append(np.uint64(key[2]))   # key[2] = merged m0
                    used[a] = True
                    used[b] = True
                    changed = True

        # Carry forward un-merged cubes
        for i in range(n):
            if not used[i]:
                new_m1.append(m1[i])
                new_m0.append(m0[i])

        if changed:
            m1 = np.array(new_m1, dtype=np.uint64)
            m0 = np.array(new_m0, dtype=np.uint64)

    return _remove_contained_bv(m1, m0, deadline)


# ======================================================================
# Local mid-recursion containment filter
# ======================================================================

def _local_filter(cubes: np.ndarray, deadline: float) -> np.ndarray:
    """Remove dominated cubes via bitvectors, then unpack back to uint8.

    Called when an intermediate result exceeds the dynamic filter_threshold
    to prevent the cube count from compounding multiplicatively across recursion 
    levels. For N~400-2000 and V<=64 this takes <5 ms.
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

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python Ha_Lenhart_complgen.py <path_to_file_or_folder>")
        sys.exit(1)

    targets = []
    for arg_path in sys.argv[1:]:
        if os.path.exists(arg_path):
            if os.path.isdir(arg_path):
                for f in sorted(os.listdir(arg_path)):
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
