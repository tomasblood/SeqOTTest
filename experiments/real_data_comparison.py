"""
Comprehensive comparison of alignment methods on real NeurIPS/ArXiv data

This script:
1. Loads real paper embeddings
2. Runs Global SeqOT, Procrustes, and Aligned UMAP
3. Generates comprehensive visualizations
4. Saves detailed results
"""

import numpy as np
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.seqot.data_loaders import NeurIPSDataLoader, create_sample_neurips_data
from src.seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment
from src.seqot.metrics import evaluate_alignment, flow_conservation_error, sparsity_metric
from src.seqot.visualizations import create_comprehensive_comparison_report
from src.seqot.utils import compute_cosine_distance


def run_comparison_experiment(data_path, output_dir='results/real_data',
                              epsilon=0.05, verbose=True):
    """
    Run comprehensive comparison on real data.

    Parameters
    ----------
    data_path : str
        Path to pickle file with embeddings
    output_dir : str
        Directory for results
    epsilon : float
        Regularization parameter for SeqOT
    verbose : bool
        Print progress
    """
    print("=" * 80)
    print("REAL DATA COMPARISON EXPERIMENT")
    print("=" * 80)
    print()

    # ===========================
    # 1. Load Data
    # ===========================
    print("Step 1: Loading data...")
    loader = NeurIPSDataLoader()
    loader.load_from_pickle(data_path)

    embeddings, years, metadata_by_year = loader.get_sequential_embeddings()

    print(f"\nLoaded {len(years)} years: {years}")
    for year, emb in zip(years, embeddings):
        print(f"  {year}: {emb.shape[0]} papers, {emb.shape[1]} dims")
    print()

    # ===========================
    # 2. Run Global SeqOT
    # ===========================
    print("Step 2: Running Global SeqOT...")
    print(f"  epsilon = {epsilon}")

    seqot_aligner = GlobalSeqOTAlignment(
        epsilon=epsilon,
        max_iter=1000,
        tol=1e-6,
        verbose=verbose
    )

    aligned_seqot = seqot_aligner.fit_transform(embeddings)
    couplings_seqot = seqot_aligner.get_couplings()

    print(f"  Converged: {seqot_aligner.solver_.converged_}")
    print(f"  Iterations: {seqot_aligner.solver_.n_iter_}")

    # Flow conservation check
    errors, max_error = flow_conservation_error(couplings_seqot)
    print(f"  Flow conservation error: {max_error:.2e}")

    # Sparsity
    gini_seqot, mean_gini_seqot = sparsity_metric(couplings_seqot)
    print(f"  Mean Gini coefficient: {mean_gini_seqot:.3f}")
    print()

    # ===========================
    # 3. Run Procrustes
    # ===========================
    print("Step 3: Running Procrustes alignment...")

    proc_aligner = ProcrustesAlignment(center=True, scale=True)
    aligned_proc = proc_aligner.fit_transform(embeddings)

    print("  Done")
    print()

    # ===========================
    # 4. Run Greedy Baseline
    # ===========================
    print("Step 4: Running Greedy baseline (independent OT steps)...")

    from src.seqot.sinkhorn import ForwardBackwardSinkhorn

    # Compute cost matrices
    cost_matrices = [
        compute_cosine_distance(embeddings[t], embeddings[t + 1])
        for t in range(len(embeddings) - 1)
    ]

    # Solve each step independently
    couplings_greedy = []
    for t, C in enumerate(cost_matrices):
        solver = ForwardBackwardSinkhorn(epsilon=epsilon, max_iter=1000, tol=1e-6, verbose=False)
        mu_t = np.ones(C.shape[0]) / C.shape[0]
        nu_t = np.ones(C.shape[1]) / C.shape[1]
        solver.fit([C], mu=mu_t, nu=nu_t)
        couplings_greedy.append(solver.get_couplings()[0])

    print("  Done")
    print()

    # ===========================
    # 5. Evaluate Alignments
    # ===========================
    print("Step 5: Evaluating alignment quality...")

    # For real data, we evaluate how well each method preserves the original structure
    # We use the first year as reference
    target_embeddings = [embeddings[0]] * len(embeddings)

    results_seqot = evaluate_alignment(
        embeddings,
        target_embeddings,
        aligned_seqot,
        method_name="Global SeqOT"
    )

    results_proc = evaluate_alignment(
        embeddings,
        target_embeddings,
        aligned_proc,
        method_name="Procrustes"
    )

    # Create "aligned" version for greedy (just use original embeddings)
    # since greedy doesn't produce aligned embeddings directly
    results_greedy = {
        'method': 'Greedy Baseline',
        'mean_euclidean_error': np.nan,
        'mean_cosine_distance': np.nan,
        'mean_procrustes_error': np.nan,
        'mean_correlation': np.nan,
    }

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    for results in [results_seqot, results_proc]:
        print(f"{results['method']}:")
        print(f"  Mean Euclidean Error:  {results['mean_euclidean_error']:.4f}")
        print(f"  Mean Cosine Distance:  {results['mean_cosine_distance']:.4f}")
        print(f"  Mean Procrustes Error: {results['mean_procrustes_error']:.4f}")
        print(f"  Mean Correlation:      {results['mean_correlation']:.4f}")
        print()

    # ===========================
    # 6. Analyze Transport Properties
    # ===========================
    print("Step 6: Analyzing transport properties...")
    print()

    # Compare sparsity
    gini_greedy, mean_gini_greedy = sparsity_metric(couplings_greedy)

    print("Sparsity (Gini coefficient):")
    print(f"  Global SeqOT:     {mean_gini_seqot:.4f}")
    print(f"  Greedy Baseline:  {mean_gini_greedy:.4f}")
    print()

    # Compare transport costs
    from src.seqot.metrics import compute_transport_cost

    cost_seqot = compute_transport_cost(couplings_seqot, cost_matrices)
    cost_greedy = compute_transport_cost(couplings_greedy, cost_matrices)

    print("Total Transport Cost:")
    print(f"  Global SeqOT:     {cost_seqot:.4f}")
    print(f"  Greedy Baseline:  {cost_greedy:.4f}")
    print(f"  Difference:       {cost_greedy - cost_seqot:.4f}")
    print()

    # ===========================
    # 7. Generate Visualizations
    # ===========================
    print("Step 7: Generating comprehensive visualizations...")
    print()

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data for visualization
    embeddings_dict = {
        'Global SeqOT': aligned_seqot,
        'Procrustes': aligned_proc,
        'Original': embeddings,
    }

    results_dict = {
        'Global SeqOT': results_seqot,
        'Procrustes': results_proc,
    }

    couplings_dict = {
        'Global SeqOT': couplings_seqot,
        'Greedy Baseline': couplings_greedy,
    }

    # Create metadata dict
    metadata_dict = {year: meta for year, meta in zip(years, metadata_by_year)}

    # Generate all visualizations
    figures = create_comprehensive_comparison_report(
        embeddings_dict,
        years,
        metadata_dict,
        results_dict,
        couplings_dict,
        output_dir=output_dir,
        prefix='neurips'
    )

    print(f"All visualizations saved to {output_dir}/")
    print()

    # ===========================
    # 8. Save Results
    # ===========================
    print("Step 8: Saving results...")

    results_summary = {
        'years': years,
        'n_papers_per_year': [emb.shape[0] for emb in embeddings],
        'embedding_dim': embeddings[0].shape[1],
        'epsilon': epsilon,
        'seqot': {
            'converged': seqot_aligner.solver_.converged_,
            'n_iterations': seqot_aligner.solver_.n_iter_,
            'flow_error': max_error,
            'mean_gini': mean_gini_seqot,
            'transport_cost': cost_seqot,
            **results_seqot
        },
        'procrustes': results_proc,
        'greedy': {
            'mean_gini': mean_gini_greedy,
            'transport_cost': cost_greedy,
        }
    }

    import json
    with open(f'{output_dir}/results_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"Results saved to {output_dir}/results_summary.json")
    print()

    # ===========================
    # Summary
    # ===========================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    if results_seqot['mean_euclidean_error'] < results_proc['mean_euclidean_error']:
        improvement = (
            (results_proc['mean_euclidean_error'] - results_seqot['mean_euclidean_error'])
            / results_proc['mean_euclidean_error'] * 100
        )
        print(f"✓ Global SeqOT outperforms Procrustes by {improvement:.1f}% in alignment error")
    else:
        print(f"  Procrustes achieved lower alignment error")

    if mean_gini_seqot > mean_gini_greedy:
        print(f"✓ Global SeqOT produces sparser transport ({mean_gini_seqot:.3f} vs {mean_gini_greedy:.3f})")
    else:
        print(f"  Greedy baseline produced sparser transport")

    if cost_seqot < cost_greedy:
        print(f"✓ Global SeqOT achieves lower total cost ({cost_seqot:.2f} vs {cost_greedy:.2f})")
    else:
        print(f"  Greedy baseline achieved lower cost")

    print()
    print("Experiment complete!")
    print()

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare alignment methods on real data')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to embeddings pickle file')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample NeurIPS-like data')
    parser.add_argument('--epsilon', type=float, default=0.05,
                       help='Regularization parameter')
    parser.add_argument('--output', type=str, default='results/real_data',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed progress')

    args = parser.parse_args()

    # If no data provided, create sample data
    if args.data is None or args.create_sample:
        print("Creating sample NeurIPS-like data...")
        data_path = create_sample_neurips_data(
            output_path='data/neurips/sample_embeddings.pkl',
            n_years=6,
            n_papers_per_year=200,
            n_dims=300,
            random_state=42
        )
        print()
    else:
        data_path = args.data

    # Run experiment
    results = run_comparison_experiment(
        data_path=data_path,
        output_dir=args.output,
        epsilon=args.epsilon,
        verbose=args.verbose
    )
