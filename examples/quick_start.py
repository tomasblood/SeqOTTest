"""
Quick Start Example: Global SeqOT vs Procrustes

This script demonstrates basic usage of Global SeqOT and compares it with Procrustes alignment.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment
from src.seqot.data_generators import generate_rotating_embeddings
from src.seqot.metrics import evaluate_alignment

print("=" * 70)
print("GLOBAL SEQOT - QUICK START EXAMPLE")
print("=" * 70)
print()

# Step 1: Generate synthetic data
print("Step 1: Generating rotating embeddings...")
embeddings, correspondences = generate_rotating_embeddings(
    n_steps=5,
    n_points=50,
    n_dims=10,
    rotation_angle=np.pi / 6,
    noise=0.1,
    random_state=42
)

print(f"  Created {len(embeddings)} time steps")
print(f"  Each has {embeddings[0].shape[0]} points in {embeddings[0].shape[1]} dimensions")
print()

# Step 2: Align with Global SeqOT
print("Step 2: Aligning with Global SeqOT...")
seqot_aligner = GlobalSeqOTAlignment(
    epsilon=0.05,
    max_iter=500,
    tol=1e-6,
    verbose=False
)

aligned_seqot = seqot_aligner.fit_transform(embeddings)

print(f"  Converged: {seqot_aligner.solver_.converged_}")
print(f"  Iterations: {seqot_aligner.solver_.n_iter_}")
print()

# Step 3: Align with Procrustes (baseline)
print("Step 3: Aligning with Procrustes (baseline)...")
proc_aligner = ProcrustesAlignment(center=True, scale=True)
aligned_proc = proc_aligner.fit_transform(embeddings)
print("  Done")
print()

# Step 4: Evaluate
print("Step 4: Evaluating alignment quality...")
print()

# For rotating data, we can measure how well we recovered the original structure
# by checking distance to the first embedding
target = [embeddings[0]] * len(embeddings)

results_seqot = evaluate_alignment(
    embeddings, target, aligned_seqot, method_name="Global SeqOT"
)

results_proc = evaluate_alignment(
    embeddings, target, aligned_proc, method_name="Procrustes"
)

print("-" * 70)
print("RESULTS")
print("-" * 70)
print()
print(f"Global SeqOT:")
print(f"  Mean Euclidean Error:  {results_seqot['mean_euclidean_error']:.4f}")
print(f"  Mean Cosine Distance:  {results_seqot['mean_cosine_distance']:.4f}")
print(f"  Mean Procrustes Error: {results_seqot['mean_procrustes_error']:.4f}")
print(f"  Mean Correlation:      {results_seqot['mean_correlation']:.4f}")
print()

print(f"Procrustes:")
print(f"  Mean Euclidean Error:  {results_proc['mean_euclidean_error']:.4f}")
print(f"  Mean Cosine Distance:  {results_proc['mean_cosine_distance']:.4f}")
print(f"  Mean Procrustes Error: {results_proc['mean_procrustes_error']:.4f}")
print(f"  Mean Correlation:      {results_proc['mean_correlation']:.4f}")
print()

# Step 5: Analyze transport couplings
print("Step 5: Analyzing transport structure...")
couplings = seqot_aligner.get_couplings()

print(f"  Number of couplings: {len(couplings)}")
for i, P in enumerate(couplings):
    print(f"  Coupling {i}: shape {P.shape}, sum = {P.sum():.6f}")

from src.seqot.metrics import flow_conservation_error

errors, max_error = flow_conservation_error(couplings)
print(f"\n  Flow conservation error: {max_error:.2e}")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

if results_seqot['mean_euclidean_error'] < results_proc['mean_euclidean_error']:
    improvement = (
        (results_proc['mean_euclidean_error'] - results_seqot['mean_euclidean_error'])
        / results_proc['mean_euclidean_error']
        * 100
    )
    print(f"✓ Global SeqOT outperforms Procrustes by {improvement:.1f}%!")
    print("  This demonstrates the advantage of global optimization.")
else:
    print("  Procrustes performed better on this example.")

print()
print("Next steps:")
print("  - Run: python experiments/tunneling_experiment.py")
print("  - Open: notebooks/alignment_comparison.ipynb")
print("  - Test: pytest tests/ -v")
print()
