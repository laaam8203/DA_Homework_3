import sys
import os
from espresso_parser import parse_cover

def check_scc_minimality(filepath):
    print(f"\n{'='*60}")
    print(f"  SCC-Minimality Verification: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    try:
        cover = parse_cover(filepath)
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    cubes = cover.cubes
    num_vars = cover.num_inputs
    num_cubes = len(cubes)
    
    print(f"  Cubes to check: {num_cubes}")
    
    # 1. Check for containment: Ci subset Cj
    print("  Checking for containment...")
    containment_found = False
    for i in range(num_cubes):
        ci = cubes[i]
        for j in range(num_cubes):
            if i == j: continue
            cj = cubes[j]
            # ci is contained in cj if cj has '-' where ci has '0'/'1' or ci has same literal
            # Simplified: ci is subset of cj if (ci matches everything cj requires)
            is_subset = True
            for k in range(num_vars):
                if cj[k] != '-' and cj[k] != ci[k]:
                    is_subset = False
                    break
            
            if is_subset:
                print(f"  [!] Violation: Cube {i} ({ci}) is contained in Cube {j} ({cj})")
                containment_found = True
                break # Only need one violation to fail SCC
                
    if not containment_found:
        print("  PASS: No single-cube containment found.")

    # 2. Check for adjacency: distance-1 merges
    print("  Checking for adjacency (merging)...")
    merges_found = False
    for i in range(num_cubes):
        ci = cubes[i]
        for j in range(i + 1, num_cubes):
            cj = cubes[j]
            
            diff_indices = [k for k in range(num_vars) if ci[k] != cj[k]]
            if len(diff_indices) == 1:
                pos = diff_indices[0]
                if ci[pos] in '01' and cj[pos] in '01':
                    print(f"  [!] Violation: Cube {i} ({ci}) and Cube {j} ({cj}) can be merged at pos {pos}")
                    merges_found = True
                    break

    if not merges_found:
        print("  PASS: No mergeable adjacent cubes found.")

    print(f"{'='*60}")
    if not containment_found and not merges_found:
        print("  RESULT: SUCCESS. The cover is SCC-minimal.")
    else:
        print("  RESULT: FAILED. The cover is NOT SCC-minimal.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify-scc.py <path_to_compl_file>")
        sys.exit(1)
    
    for path in sys.argv[1:]:
        if os.path.exists(path):
            check_scc_minimality(path)
        else:
            print(f"File not found: {path}")
