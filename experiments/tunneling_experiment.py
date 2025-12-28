"""
Tunneling Experiment: Validate Global SeqOT vs Greedy Baseline

This experiment demonstrates the key advantage of Global SeqOT:
it can "tunnel" through semantically relevant intermediate points
rather than diffusing into irrelevant distractors.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment
from src.seqot.sinkhorn import ForwardBackwardSinkhorn
from src.seqot.data_generators import generate_tunneling_scenario
from src.seqot.metrics import tunneling_score, sparsity_metric
from src.seqot.utils import compute_cosine_distance


def run_tunneling_experiment(n_bridge=10, n_distractor=90, epsilon=0.05,
                             random_state=42):
    """
    Run the tunneling experiment.

    Parameters
    ----------
    n_bridge : int
        Number of bridge (relevant) points
    n_distractor : int
        Number of distractor points
    epsilon : float
        Regularization parameter
    random_state : int
        Random seed

    Returns
    -------
    results : dict
        Experiment results
    """
    print("=" * 60)
    print("TUNNELING EXPERIMENT")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Bridge points: {n_bridge}")
    print(f"  Distractor points: {n_distractor}")
    print(f"  Epsilon: {epsilon}")
    print()

    # Generate data
    embeddings, bridge_indices = generate_tunneling_scenario(
        n_bridge_points=n_bridge,
        n_distractor_points=n_distractor,
        n_dims=10,
        random_state=random_state
    )

    print(f"Generated {len(embeddings)} time steps:")
    for i, emb in enumerate(embeddings):
        print(f"  Step {i}: {emb.shape[0]} points, {emb.shape[1]} dims")
    print()

    # Compute cost matrices
    cost_matrices = [
        compute_cosine_distance(embeddings[t], embeddings[t + 1])
        for t in range(len(embeddings) - 1)
    ]

    # ===========================
    # Global SeqOT
    # ===========================
    print("Running Global SeqOT...")
    solver_global = ForwardBackwardSinkhorn(
        epsilon=epsilon,
        max_iter=1000,
        tol=1e-7,
        verbose=True
    )

    # Use uniform marginals
    mu = np.ones(embeddings[0].shape[0]) / embeddings[0].shape[0]
    nu = np.ones(embeddings[-1].shape[0]) / embeddings[-1].shape[0]

    solver_global.fit(cost_matrices, mu=mu, nu=nu)
    couplings_global = solver_global.get_couplings()

    # Measure tunneling
    bridge_mass_global = tunneling_score(couplings_global, bridge_indices)
    gini_global, mean_gini_global = sparsity_metric(couplings_global)

    print(f"\nGlobal SeqOT Results:")
    print(f"  Converged: {solver_global.converged_}")
    print(f"  Iterations: {solver_global.n_iter_}")
    print(f"  Bridge mass at each step: {[f'{m:.3f}' for m in bridge_mass_global]}")
    print(f"  Mean Gini coefficient: {mean_gini_global:.3f}")
    print()

    # ===========================
    # Greedy Baseline (Step-by-step Sinkhorn)
    # ===========================
    print("Running Greedy Baseline (independent steps)...")

    couplings_greedy = []

    for t, C in enumerate(cost_matrices):
        # Solve each step independently with uniform marginals
        solver_greedy = ForwardBackwardSinkhorn(
            epsilon=epsilon,
            max_iter=1000,
            tol=1e-7,
            verbose=False
        )

        mu_t = np.ones(C.shape[0]) / C.shape[0]
        nu_t = np.ones(C.shape[1]) / C.shape[1]

        solver_greedy.fit([C], mu=mu_t, nu=nu_t)
        couplings_greedy.append(solver_greedy.get_couplings()[0])

    bridge_mass_greedy = tunneling_score(couplings_greedy, bridge_indices)
    gini_greedy, mean_gini_greedy = sparsity_metric(couplings_greedy)

    print(f"Greedy Baseline Results:")
    print(f"  Bridge mass at each step: {[f'{m:.3f}' for m in bridge_mass_greedy]}")
    print(f"  Mean Gini coefficient: {mean_gini_greedy:.3f}")
    print()

    # ===========================
    # Analysis
    # ===========================
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    total_bridge_global = sum(bridge_mass_global)
    total_bridge_greedy = sum(bridge_mass_greedy)

    print(f"\nTotal mass through bridges:")
    print(f"  Global SeqOT: {total_bridge_global:.3f}")
    print(f"  Greedy Baseline: {total_bridge_greedy:.3f}")
    print(f"  Improvement: {(total_bridge_global - total_bridge_greedy):.3f} "
          f"({100 * (total_bridge_global - total_bridge_greedy) / total_bridge_greedy:.1f}%)")

    print(f"\nSparsity (Gini coefficient):")
    print(f"  Global SeqOT: {mean_gini_global:.3f}")
    print(f"  Greedy Baseline: {mean_gini_greedy:.3f}")

    if total_bridge_global > total_bridge_greedy:
        print("\n✓ SUCCESS: Global SeqOT concentrates more mass on bridge points!")
        print("  This demonstrates the 'tunneling' behavior.")
    else:
        print("\n✗ WARNING: Global SeqOT did not outperform greedy baseline.")
        print("  Consider adjusting epsilon or problem setup.")

    # Return results
    results = {
        'embeddings': embeddings,
        'bridge_indices': bridge_indices,
        'couplings_global': couplings_global,
        'couplings_greedy': couplings_greedy,
        'bridge_mass_global': bridge_mass_global,
        'bridge_mass_greedy': bridge_mass_greedy,
        'gini_global': gini_global,
        'gini_greedy': gini_greedy,
    }

    return results


def visualize_results(results, save_path=None):
    """
    Visualize the tunneling experiment results.

    Parameters
    ----------
    results : dict
        Results from run_tunneling_experiment
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Bridge mass over time
    ax = axes[0, 0]
    steps = range(len(results['bridge_mass_global']))
    ax.plot(steps, results['bridge_mass_global'], 'o-', label='Global SeqOT',
            linewidth=2, markersize=8)
    ax.plot(steps, results['bridge_mass_greedy'], 's-', label='Greedy Baseline',
            linewidth=2, markersize=8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Mass Through Bridges', fontsize=12)
    ax.set_title('Tunneling Behavior: Mass on Bridge Points', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Gini coefficients
    ax = axes[0, 1]
    steps = range(len(results['gini_global']))
    ax.plot(steps, results['gini_global'], 'o-', label='Global SeqOT',
            linewidth=2, markersize=8)
    ax.plot(steps, results['gini_greedy'], 's-', label='Greedy Baseline',
            linewidth=2, markersize=8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('Sparsity of Transport Plans', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Coupling matrix visualization for Global SeqOT (step 1)
    ax = axes[1, 0]
    im1 = ax.imshow(results['couplings_global'][1], cmap='hot', aspect='auto')
    ax.set_title('Global SeqOT Coupling (Step 1→2)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Points', fontsize=11)
    ax.set_ylabel('Source Points', fontsize=11)
    plt.colorbar(im1, ax=ax)

    # Mark bridge points
    n_bridge = len(results['bridge_indices'][1])
    ax.axvline(n_bridge - 0.5, color='cyan', linestyle='--', linewidth=2,
               label='Bridge boundary')
    ax.legend(fontsize=9)

    # Plot 4: Coupling matrix visualization for Greedy (step 1)
    ax = axes[1, 1]
    im2 = ax.imshow(results['couplings_greedy'][1], cmap='hot', aspect='auto')
    ax.set_title('Greedy Baseline Coupling (Step 1→2)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Points', fontsize=11)
    ax.set_ylabel('Source Points', fontsize=11)
    plt.colorbar(im2, ax=ax)

    # Mark bridge points
    ax.axvline(n_bridge - 0.5, color='cyan', linestyle='--', linewidth=2,
               label='Bridge boundary')
    ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    return fig


if __name__ == "__main__":
    # Run experiment
    results = run_tunneling_experiment(
        n_bridge=10,
        n_distractor=90,
        epsilon=0.05,
        random_state=42
    )

    # Visualize
    fig = visualize_results(results, save_path='tunneling_results.png')
    plt.show()

    print("\nExperiment complete!")
