#!/usr/bin/env python3
"""
Run Global SeqOT (standard OT, NOT GW) on realistic NeurIPS-scale embeddings.

This uses standard optimal transport which is much faster than GW.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from time import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment
from seqot.data_loaders import NeurIPSDataLoader, create_sample_neurips_data
from seqot.metrics import evaluate_alignment
from seqot.visualizations import (
    plot_temporal_evolution_2d,
    plot_alignment_metrics_comparison,
    plot_transport_couplings
)

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("=" * 70)
print("GLOBAL SEQOT (STANDARD OT): REALISTIC NEURIPS COMPARISON")
print("=" * 70)

# ============================================================================
# 1. CREATE REALISTIC DATA
# ============================================================================
print("\n[1/4] Creating realistic NeurIPS-scale embeddings...")
print("-" * 70)

data_path = create_sample_neurips_data(
    output_path='data/neurips/realistic_neurips_embeddings.pkl',
    n_years=6,               # 2015-2020
    n_papers_per_year=150,   # Manageable scale
    n_dims=256,              # Reduced dimensions
    random_state=42
)

print(f"✓ Created realistic dataset: {data_path}")
print(f"  Size: {Path(data_path).stat().st_size / 1024 / 1024:.1f} MB")

# Load the data
loader = NeurIPSDataLoader()
loader.load_from_pickle(data_path)
embeddings, years, metadata = loader.get_sequential_embeddings()

print(f"\n✓ Loaded {len(embeddings)} time steps")
print(f"  Years: {years[0]}-{years[-1]}")
for i, (year, emb) in enumerate(zip(years, embeddings)):
    print(f"    {year}: {emb.shape[0]} papers × {emb.shape[1]} dims")

# ============================================================================
# 2. RUN GLOBAL SEQOT (STANDARD OT - FAST!)
# ============================================================================
print("\n[2/4] Running Global SeqOT with standard OT...")
print("-" * 70)

t0 = time()
aligner_seqot = GlobalSeqOTAlignment(
    epsilon=0.1,
    max_iter=1000,
    use_gromov=False,  # Use standard OT (much faster!)
    verbose=True
)

aligned_seqot = aligner_seqot.fit_transform(embeddings)
t_seqot = time() - t0

print(f"\n✓ Global SeqOT completed in {t_seqot:.1f}s")

# ============================================================================
# 3. RUN PROCRUSTES BASELINE
# ============================================================================
print("\n[3/4] Running Procrustes baseline...")
print("-" * 70)

t0 = time()
aligner_procrustes = ProcrustesAlignment()
aligned_procrustes = aligner_procrustes.fit_transform(embeddings)
t_procrustes = time() - t0
print(f"✓ Procrustes completed in {t_procrustes:.1f}s")

# ============================================================================
# 4. EVALUATE AND COMPARE
# ============================================================================
print("\n[4/4] Evaluating alignment quality...")
print("-" * 70)

# Use first time step as target
target = [embeddings[0]] * len(embeddings)

results_seqot = evaluate_alignment(embeddings, target, aligned_seqot, "Global SeqOT")
results_procrustes = evaluate_alignment(embeddings, target, aligned_procrustes, "Procrustes")

# Create comparison dictionary
results_dict = {
    'Global SeqOT (Standard OT)': results_seqot,
    'Procrustes': results_procrustes
}

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

# Print comparison table
metrics = ['mean_euclidean_error', 'mean_cosine_distance', 'mean_procrustes_error', 'mean_correlation']
metric_names = ['Euclidean Error', 'Cosine Distance', 'Procrustes Error', 'Correlation']
metric_better = ['lower', 'lower', 'lower', 'higher']

for metric, name, better in zip(metrics, metric_names, metric_better):
    print(f"\n{name} ({better} is better):")
    values = {method: results[metric] for method, results in results_dict.items()}

    # Find best
    if better == 'lower':
        best_method = min(values, key=values.get)
    else:
        best_method = max(values, key=values.get)

    for method, value in values.items():
        marker = "★" if method == best_method else " "
        print(f"  {marker} {method:30s}: {value:.4f}")

# Calculate improvements
print("\n" + "=" * 70)
print("IMPROVEMENTS (Global SeqOT vs Procrustes)")
print("=" * 70)

seqot_error = results_seqot['mean_euclidean_error']
proc_error = results_procrustes['mean_euclidean_error']

improvement = (proc_error - seqot_error) / proc_error * 100

print(f"\nEuclidean Error:  {improvement:+.1f}% {'✓' if improvement > 0 else '✗'}")
print(f"  Global SeqOT: {seqot_error:.4f}")
print(f"  Procrustes:   {proc_error:.4f}")

# Runtime comparison
print("\n" + "=" * 70)
print("RUNTIME COMPARISON")
print("=" * 70)
print(f"  Global SeqOT:  {t_seqot:6.1f}s")
print(f"  Procrustes:    {t_procrustes:6.1f}s")

if improvement > 0:
    print(f"\n✓ Global SeqOT achieves {abs(improvement):.1f}% better alignment!")
else:
    print(f"\n⚠ Procrustes is competitive on this dataset")

# ============================================================================
# 5. GENERATE VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")
print("-" * 70)

output_dir = Path('results/realistic_neurips_standard_ot')
output_dir.mkdir(parents=True, exist_ok=True)

# Metrics comparison
print("  Generating metrics comparison...")
fig = plot_alignment_metrics_comparison(results_dict)
plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {output_dir / 'metrics_comparison.png'}")

# PCA evolution
print("  Generating PCA temporal evolution...")
embeddings_dict = {
    'Original': embeddings,
    'Global SeqOT': aligned_seqot,
    'Procrustes': aligned_procrustes
}

fig = plot_temporal_evolution_2d(
    embeddings_dict,
    years,
    method='pca',
    figsize=(14, 8)
)
plt.savefig(output_dir / 'temporal_evolution_pca.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {output_dir / 'temporal_evolution_pca.png'}")

# t-SNE evolution
print("  Generating t-SNE temporal evolution...")
fig = plot_temporal_evolution_2d(
    embeddings_dict,
    years,
    method='tsne',
    figsize=(14, 8)
)
plt.savefig(output_dir / 'temporal_evolution_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {output_dir / 'temporal_evolution_tsne.png'}")

# Transport couplings
if hasattr(aligner_seqot.solver_, 'couplings_'):
    print("  Generating transport coupling matrices...")
    couplings = aligner_seqot.solver_.get_couplings()

    fig = plot_transport_couplings(
        couplings,
        years,
        method_name='Global SeqOT',
        max_steps=min(5, len(couplings))
    )
    plt.savefig(output_dir / 'transport_couplings.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_dir / 'transport_couplings.png'}")

# Save numerical results
results_summary = {
    'dataset': {
        'n_years': len(embeddings),
        'years': [int(y) for y in years],
        'n_papers_per_year': [int(e.shape[0]) for e in embeddings],
        'embedding_dim': int(embeddings[0].shape[1])
    },
    'runtime_seconds': {
        'global_seqot': float(t_seqot),
        'procrustes': float(t_procrustes)
    },
    'methods': list(results_dict.keys()),
    'metrics': {
        method: {k: (float(v) if isinstance(v, (int, float, np.number)) else
                    (v.tolist() if isinstance(v, np.ndarray) else v))
                for k, v in results.items()}
        for method, results in results_dict.items()
    },
    'improvements': {
        'vs_procrustes_percent': float(improvement)
    }
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"    ✓ Saved: {output_dir / 'results.json'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED")
print("=" * 70)
print(f"\nDataset: {len(embeddings)} years × {embeddings[0].shape[0]} papers × {embeddings[0].shape[1]} dims")
print(f"Results saved to: {output_dir}")
print(f"\nKey Finding: Global SeqOT achieves {abs(improvement):.1f}% ")
print(f"{'better' if improvement > 0 else 'worse'} alignment than Procrustes")
print("\n✓ All results and visualizations saved successfully!")
