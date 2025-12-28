"""
Tests for alignment methods
"""

import numpy as np
import pytest
from src.seqot.alignment import GlobalSeqOTAlignment, ProcrustesAlignment
from src.seqot.data_generators import (
    generate_rotating_embeddings,
    generate_concept_drift,
    generate_varying_size_embeddings
)


def test_global_seqot_alignment():
    """Test Global SeqOT alignment on rotating embeddings."""
    embeddings, _ = generate_rotating_embeddings(
        n_steps=4,
        n_points=30,
        n_dims=2,
        rotation_angle=np.pi/8,
        noise=0.05
    )

    aligner = GlobalSeqOTAlignment(epsilon=0.05, max_iter=500, tol=1e-6)
    aligned = aligner.fit_transform(embeddings)

    assert len(aligned) == len(embeddings), "Wrong number of aligned embeddings"

    # Check that alignments have correct shapes
    for i, (orig, align) in enumerate(zip(embeddings, aligned)):
        assert align.shape == orig.shape, f"Shape mismatch at step {i}"


def test_procrustes_alignment():
    """Test Procrustes alignment."""
    embeddings, _ = generate_rotating_embeddings(
        n_steps=4,
        n_points=30,
        n_dims=2,
        rotation_angle=np.pi/8,
        noise=0.05
    )

    aligner = ProcrustesAlignment(center=True, scale=True)
    aligned = aligner.fit_transform(embeddings)

    assert len(aligned) == len(embeddings), "Wrong number of aligned embeddings"

    for i, align in enumerate(aligned):
        assert np.isfinite(align).all(), f"Non-finite values at step {i}"


def test_alignment_with_concept_drift():
    """Test alignment on data with concept drift."""
    embeddings, _ = generate_concept_drift(
        n_steps=5,
        n_points_per_cluster=20,
        n_dims=10,
        drift_rate=0.3
    )

    # Test Global SeqOT
    aligner_seqot = GlobalSeqOTAlignment(epsilon=0.1, max_iter=500)
    aligned_seqot = aligner_seqot.fit_transform(embeddings)

    assert len(aligned_seqot) == len(embeddings)

    # Test Procrustes
    aligner_proc = ProcrustesAlignment()
    aligned_proc = aligner_proc.fit_transform(embeddings)

    assert len(aligned_proc) == len(embeddings)


def test_alignment_varying_sizes():
    """Test alignment with varying number of points."""
    embeddings = generate_varying_size_embeddings(
        sizes=[30, 50, 40, 60],
        n_dims=8,
        noise=0.1
    )

    aligner = GlobalSeqOTAlignment(epsilon=0.1, max_iter=500)
    aligned = aligner.fit_transform(embeddings)

    assert len(aligned) == len(embeddings)

    # Check that sizes are preserved (after transport)
    for i, align in enumerate(aligned):
        assert align.shape[0] == embeddings[0].shape[0], \
            f"First dimension should match source at step {i}"


def test_get_couplings():
    """Test that we can retrieve couplings after fitting."""
    embeddings, _ = generate_rotating_embeddings(n_steps=3, n_points=20)

    aligner = GlobalSeqOTAlignment(epsilon=0.1)
    aligner.fit_transform(embeddings)

    couplings = aligner.get_couplings()

    assert len(couplings) == len(embeddings) - 1, "Wrong number of couplings"

    for i, P in enumerate(couplings):
        assert P.shape[0] == embeddings[i].shape[0], f"Wrong source size for coupling {i}"
        assert P.shape[1] == embeddings[i+1].shape[0], f"Wrong target size for coupling {i}"
        assert np.abs(P.sum() - 1.0) < 1e-4, f"Coupling {i} doesn't sum to 1"


def test_intermediate_distributions():
    """Test that we can get intermediate distributions."""
    embeddings, _ = generate_rotating_embeddings(n_steps=4, n_points=25)

    aligner = GlobalSeqOTAlignment(epsilon=0.1)
    aligner.fit_transform(embeddings)

    distributions = aligner.get_intermediate_distributions()

    assert len(distributions) == len(embeddings), "Wrong number of distributions"

    for i, dist in enumerate(distributions):
        assert len(dist) == embeddings[i].shape[0], f"Wrong size for distribution {i}"
        assert np.abs(dist.sum() - 1.0) < 1e-4, f"Distribution {i} doesn't sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
