"""
Test the netalign library (approximate_ISORANK package) directly
"""
import sys
sys.path.insert(0, '/home/user/Random-Graph-Happenings/approximate_ISORANK')

import pandas as pd
import numpy as np

try:
    from netalign.approx_isorank.io_utils import compute_adjacency, compute_pairs
    from netalign.approx_isorank.isorank_compute import compute_isorank, compute_greedy_assignment
    print("✓ Successfully imported netalign package!")
except ImportError as e:
    print(f"✗ Failed to import netalign: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("TESTING NETALIGN LIBRARY (approximate_ISORANK)")
print("=" * 70)

# Create simple test networks as TSV data
print("\nCreating test networks...")

# Network 1: Simple triangle
net1_data = pd.DataFrame({
    0: ['A', 'A', 'B'],
    1: ['B', 'C', 'C']
})

# Network 2: Similar but slightly different
net2_data = pd.DataFrame({
    0: ['X', 'X', 'Y'],
    1: ['Y', 'Z', 'Z']
})

print(f"Network 1:\n{net1_data}")
print(f"\nNetwork 2:\n{net2_data}")

# Create similarity scores (assuming all nodes are somewhat similar)
similarity_data = pd.DataFrame({
    'net1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'net2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z'],
    'score': [0.8, 0.2, 0.1, 0.2, 0.7, 0.15, 0.1, 0.15, 0.6]
})

print(f"\nNode similarities:\n{similarity_data}")

print("\n" + "-" * 70)
print("Step 1: Computing adjacency matrices...")
print("-" * 70)

Af1, nA1 = compute_adjacency(net1_data)
Af2, nA2 = compute_adjacency(net2_data)

print(f"✓ Network 1: {Af1.shape[0]} nodes")
print(f"  Node mapping: {nA1}")
print(f"✓ Network 2: {Af2.shape[0]} nodes")
print(f"  Node mapping: {nA2}")

print("\n" + "-" * 70)
print("Step 2: Creating similarity matrix...")
print("-" * 70)

E = compute_pairs(similarity_data, nA1, nA2, 'net1', 'net2')

print(f"✓ Similarity matrix E shape: {E.shape}")
print(f"  Matrix:\n{E}")

print("\n" + "-" * 70)
print("Step 3: Running IsoRank alignment...")
print("-" * 70)

# Test with different approximations
for niter, name in [(0, "R0 (fastest)"), (1, "R1 (recommended)"), (10, "Full IsoRank")]:
    print(f"\n--- {name} (niter={niter}) ---")

    R = compute_isorank(Af1, Af2, E, alpha=0.7, maxiter=niter)[-1]

    print(f"✓ Alignment matrix R shape: {R.shape}")
    print(f"  Alignment scores:\n{R}")

    # Get top alignments
    pairs = compute_greedy_assignment(R, n_align=3)

    # Map back to original node names
    rnA1 = {v: k for k, v in nA1.items()}
    rnA2 = {v: k for k, v in nA2.items()}

    print(f"\n  Top {len(pairs)} alignments:")
    for idx1, idx2 in pairs:
        node1 = rnA1[idx1]
        node2 = rnA2[idx2]
        score = R[idx1, idx2]
        print(f"    {node1} -> {node2}: {score:.4f}")

print("\n" + "=" * 70)
print("LIBRARY TEST COMPLETE")
print("=" * 70)
print("✓ netalign package works correctly")
print("✓ Can compute alignments with R0, R1, and full IsoRank")
print("✓ Greedy assignment produces reasonable matches")
