"""
Comprehensive visualizations for comparing alignment methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec


def plot_temporal_evolution_2d(embeddings_dict, years, method='pca',
                                title_prefix='', figsize=(16, 10),
                                n_samples=500, random_state=42):
    """
    Plot 2D temporal evolution of embeddings for multiple methods.

    Parameters
    ----------
    embeddings_dict : dict
        {'Method Name': [embeddings_year1, embeddings_year2, ...], ...}
    years : list
        List of years
    method : str
        'pca' or 'tsne' for dimensionality reduction
    title_prefix : str
        Prefix for the plot title
    figsize : tuple
        Figure size
    n_samples : int
        Max samples per year for visualization
    random_state : int
        Random seed
    """
    rng = np.random.RandomState(random_state)
    n_methods = len(embeddings_dict)
    n_years = len(years)

    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axes = [axes]

    # Color map for years
    colors = plt.cm.viridis(np.linspace(0, 1, n_years))

    for ax, (method_name, embeddings) in zip(axes, embeddings_dict.items()):
        # Collect samples from all years
        all_points = []
        all_years_labels = []

        for year_idx, (year, emb) in enumerate(zip(years, embeddings)):
            # Sample if too many points
            if emb.shape[0] > n_samples:
                indices = rng.choice(emb.shape[0], n_samples, replace=False)
                emb_sampled = emb[indices]
            else:
                emb_sampled = emb

            all_points.append(emb_sampled)
            all_years_labels.extend([year_idx] * emb_sampled.shape[0])

        all_points = np.vstack(all_points)
        all_years_labels = np.array(all_years_labels)

        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=random_state)
            points_2d = reducer.fit_transform(all_points)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
            points_2d = reducer.fit_transform(all_points)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Plot each year
        for year_idx, (year, color) in enumerate(zip(years, colors)):
            mask = all_years_labels == year_idx
            ax.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                c=[color],
                label=str(year),
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

        # Add trajectory connecting centroids
        centroids = []
        for year_idx in range(n_years):
            mask = all_years_labels == year_idx
            centroid = points_2d[mask].mean(axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)
        ax.plot(centroids[:, 0], centroids[:, 1], 'k--', linewidth=2,
                alpha=0.5, zorder=1000, label='Trajectory')

        # Annotate start and end
        ax.scatter(centroids[0, 0], centroids[0, 1], c='green', s=200,
                  marker='*', edgecolors='black', linewidths=2,
                  zorder=2000, label='Start')
        ax.scatter(centroids[-1, 0], centroids[-1, 1], c='red', s=200,
                  marker='*', edgecolors='black', linewidths=2,
                  zorder=2000, label='End')

        ax.set_title(f'{method_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
        ax.legend(loc='best', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{title_prefix}Temporal Evolution of Embeddings ({method.upper()})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_alignment_metrics_comparison(results_dict, figsize=(14, 10)):
    """
    Plot comparison of alignment metrics across methods.

    Parameters
    ----------
    results_dict : dict
        {'Method Name': results_from_evaluate_alignment, ...}
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    methods = list(results_dict.keys())
    n_methods = len(methods)

    # Extract metrics
    metrics_to_plot = [
        ('mean_euclidean_error', 'Mean Euclidean Error', 'lower'),
        ('mean_cosine_distance', 'Mean Cosine Distance', 'lower'),
        ('mean_procrustes_error', 'Mean Procrustes Error', 'lower'),
        ('mean_correlation', 'Mean Correlation', 'higher'),
    ]

    for idx, (metric_key, metric_name, better) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        values = [results_dict[m][metric_key] for m in methods]

        # Bar plot
        bars = ax.bar(range(n_methods), values, color=plt.cm.Set2(range(n_methods)))

        # Highlight best
        if better == 'lower':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9)

    plt.suptitle('Alignment Quality Comparison', fontsize=16, fontweight='bold')

    return fig


def plot_transport_couplings(couplings, years, method_name='Global SeqOT',
                             figsize=(16, 4), max_steps=4):
    """
    Visualize transport coupling matrices.

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices
    years : list
        Years corresponding to couplings
    method_name : str
        Name of the method
    figsize : tuple
        Figure size
    max_steps : int
        Maximum number of steps to show
    """
    n_couplings = min(len(couplings), max_steps)

    fig, axes = plt.subplots(1, n_couplings, figsize=figsize)
    if n_couplings == 1:
        axes = [axes]

    for i, (ax, P) in enumerate(zip(axes, couplings[:n_couplings])):
        im = ax.imshow(P, cmap='hot', aspect='auto', interpolation='nearest')
        ax.set_title(f'{years[i]} → {years[i+1]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Papers', fontsize=10)
        ax.set_ylabel('Source Papers', fontsize=10)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add statistics
        sparsity = np.sum(P > 0.001) / P.size
        max_val = np.max(P)
        ax.text(0.02, 0.98, f'Sparsity: {sparsity:.2%}\nMax: {max_val:.4f}',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{method_name}: Transport Couplings', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_flow_conservation(couplings, years, figsize=(10, 6)):
    """
    Plot flow conservation errors over time.

    Parameters
    ----------
    couplings : list of np.ndarray
        Transport matrices
    years : list
        Years
    figsize : tuple
        Figure size
    """
    from .metrics import flow_conservation_error

    errors, max_error = flow_conservation_error(couplings)

    fig, ax = plt.subplots(figsize=figsize)

    year_labels = [f'{years[i]}→{years[i+1]}' for i in range(len(errors))]

    ax.bar(range(len(errors)), errors, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels(year_labels, rotation=45, ha='right')
    ax.set_ylabel('Flow Conservation Error (L1)', fontsize=12)
    ax.set_title('Flow Conservation Errors', fontsize=14, fontweight='bold')
    ax.axhline(y=1e-5, color='red', linestyle='--', linewidth=2,
              label='Tolerance (1e-5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Add max error annotation
    ax.text(0.02, 0.98, f'Max Error: {max_error:.2e}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()

    return fig


def plot_topic_trajectories(embeddings_dict, years, topic_keywords,
                           metadata_by_year, figsize=(14, 8)):
    """
    Track specific topics (e.g., "transformers", "GANs") over time.

    Parameters
    ----------
    embeddings_dict : dict
        {'Method Name': [embeddings], ...}
    years : list
        Years
    topic_keywords : list of str
        Keywords to search for in paper titles/topics
    metadata_by_year : dict
        {year: [{'title': ..., 'topic': ...}, ...], ...}
    figsize : tuple
        Figure size
    """
    n_methods = len(embeddings_dict)
    n_topics = len(topic_keywords)

    fig, axes = plt.subplots(n_topics, n_methods, figsize=figsize,
                            sharex=True, sharey='row')

    if n_topics == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)

    for topic_idx, keyword in enumerate(topic_keywords):
        for method_idx, (method_name, embeddings) in enumerate(embeddings_dict.items()):
            ax = axes[topic_idx, method_idx]

            # Find papers matching keyword
            topic_prevalence = []
            topic_centroids = []

            for year_idx, (year, emb) in enumerate(zip(years, embeddings)):
                metadata = metadata_by_year.get(year, [])

                # Find matching papers
                matching_indices = [
                    i for i, meta in enumerate(metadata)
                    if keyword.lower() in meta.get('title', '').lower() or
                       keyword.lower() in meta.get('topic', '').lower()
                ]

                prevalence = len(matching_indices) / len(metadata) if metadata else 0
                topic_prevalence.append(prevalence)

                # Compute centroid if papers exist
                if matching_indices and len(matching_indices) < emb.shape[0]:
                    centroid = emb[matching_indices].mean(axis=0)
                    topic_centroids.append(centroid)
                else:
                    topic_centroids.append(None)

            # Plot prevalence
            ax.plot(years, topic_prevalence, 'o-', linewidth=2, markersize=8,
                   color=plt.cm.Set1(topic_idx), label=f'{keyword} prevalence')
            ax.fill_between(years, 0, topic_prevalence, alpha=0.3,
                           color=plt.cm.Set1(topic_idx))

            ax.set_ylabel('Prevalence', fontsize=10)
            if topic_idx == 0:
                ax.set_title(method_name, fontsize=12, fontweight='bold')
            if method_idx == 0:
                ax.text(-0.3, 0.5, keyword, transform=ax.transAxes,
                       fontsize=12, fontweight='bold', rotation=90,
                       verticalalignment='center')

            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(topic_prevalence) * 1.2 if topic_prevalence else 1)

    for ax in axes[-1]:
        ax.set_xlabel('Year', fontsize=11)

    plt.suptitle('Topic Trajectories Over Time', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


def create_comprehensive_comparison_report(embeddings_dict, years,
                                          metadata_by_year, results_dict,
                                          couplings_dict=None,
                                          output_dir='results',
                                          prefix='comparison'):
    """
    Create a comprehensive comparison report with all visualizations.

    Parameters
    ----------
    embeddings_dict : dict
        {'Method Name': [embeddings], ...}
    years : list
        Years
    metadata_by_year : dict
        Metadata for each year
    results_dict : dict
        Results from evaluate_alignment for each method
    couplings_dict : dict, optional
        {'Method Name': [couplings], ...}
    output_dir : str
        Directory to save figures
    prefix : str
        Prefix for output files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # 1. Temporal evolution (PCA)
    print("Creating PCA temporal evolution plot...")
    fig = plot_temporal_evolution_2d(embeddings_dict, years, method='pca',
                                     title_prefix=f'{prefix}: ')
    fig.savefig(f'{output_dir}/{prefix}_temporal_pca.png', dpi=300, bbox_inches='tight')
    figures['temporal_pca'] = fig
    plt.close(fig)

    # 2. Temporal evolution (t-SNE) - if not too many points
    total_points = sum(emb[0].shape[0] for emb in embeddings_dict.values())
    if total_points < 5000:
        print("Creating t-SNE temporal evolution plot...")
        fig = plot_temporal_evolution_2d(embeddings_dict, years, method='tsne',
                                         title_prefix=f'{prefix}: ')
        fig.savefig(f'{output_dir}/{prefix}_temporal_tsne.png', dpi=300, bbox_inches='tight')
        figures['temporal_tsne'] = fig
        plt.close(fig)

    # 3. Alignment metrics comparison
    print("Creating alignment metrics comparison...")
    fig = plot_alignment_metrics_comparison(results_dict)
    fig.savefig(f'{output_dir}/{prefix}_metrics.png', dpi=300, bbox_inches='tight')
    figures['metrics'] = fig
    plt.close(fig)

    # 4. Transport couplings (if available)
    if couplings_dict:
        for method_name, couplings in couplings_dict.items():
            print(f"Creating coupling visualization for {method_name}...")
            fig = plot_transport_couplings(couplings, years, method_name=method_name)
            fig.savefig(f'{output_dir}/{prefix}_couplings_{method_name.replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            figures[f'couplings_{method_name}'] = fig
            plt.close(fig)

            # Flow conservation
            print(f"Creating flow conservation plot for {method_name}...")
            fig = plot_flow_conservation(couplings, years)
            fig.savefig(f'{output_dir}/{prefix}_flow_{method_name.replace(" ", "_")}.png',
                       dpi=300, bbox_inches='tight')
            figures[f'flow_{method_name}'] = fig
            plt.close(fig)

    # 5. Topic trajectories (if metadata available)
    if metadata_by_year:
        # Extract common topics from metadata
        all_topics = set()
        for meta_list in metadata_by_year.values():
            for meta in meta_list:
                topic = meta.get('topic', '')
                if topic:
                    all_topics.add(topic)

        if all_topics:
            topic_keywords = list(all_topics)[:3]  # Top 3 topics
            print(f"Creating topic trajectories for: {topic_keywords}...")
            fig = plot_topic_trajectories(embeddings_dict, years, topic_keywords,
                                         metadata_by_year)
            fig.savefig(f'{output_dir}/{prefix}_topics.png', dpi=300, bbox_inches='tight')
            figures['topics'] = fig
            plt.close(fig)

    print(f"\nAll figures saved to {output_dir}/")

    return figures
