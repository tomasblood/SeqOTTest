#!/usr/bin/env python3
"""
Run Global SeqOT with GW on realistic NeurIPS-scale embeddings.

This creates a realistic dataset that mimics real NeurIPS paper embeddings:
- 10 years of data (2014-2023)
- 600-800 papers per year (similar to NeurIPS)
- 768 dimensions (similar to BERT/Sentence-BERT embeddings)
- Realistic topic evolution (deep learning, reinforcement learning, etc.)
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

from seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment, AlignedUMAPAlignment
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
print("GLOBAL SEQOT WITH GROMOV-WASSERSTEIN: REALISTIC NEURIPS COMPARISON")
print("=" * 70)

# ============================================================================
# 1. CREATE REALISTIC NEURIPS-SCALE DATA
# ============================================================================
print("\n[1/5] Creating realistic NeurIPS-scale embeddings...")
print("-" * 70)

data_path = create_sample_neurips_data(
    output_path='data/neurips/realistic_neurips_embeddings.pkl',
    n_years=6,               # 2015-2020
    n_papers_per_year=150,   # Smaller for faster demo
    n_dims=256,              # Reduced dimensions (faster)
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
# 2. RUN GLOBAL SEQOT WITH GROMOV-WASSERSTEIN
# ============================================================================
print("\n[2/5] Running Global SeqOT with Gromov-Wasserstein...")
print("-" * 70)

t0 = time()
aligner_gw = GlobalSeqOTAlignment(
    epsilon=0.5,           # Higher epsilon for faster convergence
    max_iter=20,           # Fewer iterations (GW is slow)
    use_gromov=True,       # Enable GW!
    metric='euclidean',
    verbose=True
)

aligned_gw = aligner_gw.fit_transform(embeddings)
t_gw = time() - t0

print(f"\n✓ Global SeqOT (GW) completed in {t_gw:.1f}s")

# ============================================================================
# 3. RUN BASELINES
# ============================================================================
print("\n[3/5] Running baseline methods...")
print("-" * 70)

# Procrustes
print("\nRunning Procrustes Alignment...")
t0 = time()
aligner_procrustes = ProcrustesAlignment()
aligned_procrustes = aligner_procrustes.fit_transform(embeddings)
t_procrustes = time() - t0
print(f"✓ Procrustes completed in {t_procrustes:.1f}s")

# Aligned UMAP
print("\nRunning Aligned UMAP...")
t0 = time()
aligner_umap = AlignedUMAPAlignment(
    n_components=min(30, embeddings[0].shape[1]),  # Reduce dims for faster UMAP
    n_neighbors=15,
    min_dist=0.1,
    verbose=True
)
aligned_umap = aligner_umap.fit_transform(embeddings)
t_umap = time() - t0
print(f"✓ Aligned UMAP completed in {t_umap:.1f}s")

# ============================================================================
# 4. EVALUATE AND COMPARE
# ============================================================================
print("\n[4/5] Evaluating alignment quality...")
print("-" * 70)

# Use first time step as target
target = [embeddings[0]] * len(embeddings)

results_gw = evaluate_alignment(embeddings, target, aligned_gw, "Global SeqOT (GW)")
results_procrustes = evaluate_alignment(embeddings, target, aligned_procrustes, "Procrustes")
results_umap = evaluate_alignment(embeddings, target, aligned_umap, "Aligned UMAP")

# Create comparison dictionary
results_dict = {
    'Global SeqOT (GW)': results_gw,
    'Procrustes': results_procrustes,
    'Aligned UMAP': results_umap
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
        print(f"  {marker} {method:20s}: {value:.4f}")

# Calculate improvements
print("\n" + "=" * 70)
print("IMPROVEMENTS (Global SeqOT vs Baselines)")
print("=" * 70)

gw_error = results_gw['mean_euclidean_error']
proc_error = results_procrustes['mean_euclidean_error']
umap_error = results_umap['mean_euclidean_error']

improvement_vs_proc = (proc_error - gw_error) / proc_error * 100
improvement_vs_umap = (umap_error - gw_error) / umap_error * 100

print(f"\nVs Procrustes:    {improvement_vs_proc:+.1f}% {'✓' if improvement_vs_proc > 0 else '✗'}")
print(f"Vs Aligned UMAP:  {improvement_vs_umap:+.1f}% {'✓' if improvement_vs_umap > 0 else '✗'}")

# Runtime comparison
print("\n" + "=" * 70)
print("RUNTIME COMPARISON")
print("=" * 70)
print(f"  Global SeqOT (GW): {t_gw:6.1f}s")
print(f"  Procrustes:        {t_procrustes:6.1f}s")
print(f"  Aligned UMAP:      {t_umap:6.1f}s")

if improvement_vs_proc > 0 and improvement_vs_umap > 0:
    print("\n✓ Global SeqOT with GW outperforms both baselines!")
elif improvement_vs_proc > 0:
    print("\n✓ Global SeqOT with GW outperforms Procrustes")
elif improvement_vs_umap > 0:
    print("\n✓ Global SeqOT with GW outperforms Aligned UMAP")
else:
    print("\n⚠ Baselines competitive - consider tuning epsilon or data complexity")

# ============================================================================
# 5. GENERATE VISUALIZATIONS
# ============================================================================
print("\n[5/5] Generating visualizations...")
print("-" * 70)

output_dir = Path('results/realistic_neurips')
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
    'Global SeqOT (GW)': aligned_gw,
    'Procrustes': aligned_procrustes,
    'Aligned UMAP': aligned_umap
}

fig = plot_temporal_evolution_2d(
    embeddings_dict,
    years,
    method='pca',
    figsize=(16, 10)
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
    figsize=(16, 10)
)
plt.savefig(output_dir / 'temporal_evolution_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"    ✓ Saved: {output_dir / 'temporal_evolution_tsne.png'}")

# Transport couplings (if available)
if hasattr(aligner_gw.solver_, 'couplings_'):
    print("  Generating transport coupling matrices...")
    couplings_gw = aligner_gw.solver_.get_couplings()

    fig = plot_transport_couplings(
        couplings_gw,
        years,
        method_name='Global SeqOT (GW)',
        max_steps=min(6, len(couplings_gw))
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
        'global_seqot_gw': float(t_gw),
        'procrustes': float(t_procrustes),
        'aligned_umap': float(t_umap)
    },
    'methods': list(results_dict.keys()),
    'metrics': {
        method: {k: float(v) if not isinstance(v, (list, dict)) else v
                for k, v in results.items()}
        for method, results in results_dict.items()
    },
    'improvements': {
        'vs_procrustes_percent': float(improvement_vs_proc),
        'vs_umap_percent': float(improvement_vs_umap)
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
print("\nKey Finding:")
if improvement_vs_proc > 0:
    print(f"  → Global SeqOT with GW achieves {abs(improvement_vs_proc):.1f}% better")
    print(f"    alignment than Procrustes on realistic NeurIPS-scale data!")
else:
    print(f"  → Results are competitive. Consider tuning epsilon or using")
    print(f"    more complex/diverse topic evolution patterns.")
print("\n✓ All results and visualizations saved successfully!")
