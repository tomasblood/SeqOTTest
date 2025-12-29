"""
Global SeqOT: Sequential Optimal Transport with Global Optimization
Based on Watanabe & Isobe (2024/2025) - arXiv:2412.03120
"""

from .sinkhorn import ForwardBackwardSinkhorn
from .alignment import GlobalSeqOTAlignment, ProcrustesAlignment, AlignedUMAPAlignment
from .metrics import evaluate_alignment, flow_conservation_error
from .utils import compute_cosine_distance
from .data_loaders import NeurIPSDataLoader, ArXivDataLoader, create_sample_neurips_data
from .visualizations import (
    plot_temporal_evolution_2d,
    plot_alignment_metrics_comparison,
    plot_transport_couplings,
    create_comprehensive_comparison_report
)

__version__ = "2.0.0"
__all__ = [
    "ForwardBackwardSinkhorn",
    "GlobalSeqOTAlignment",
    "ProcrustesAlignment",
    "AlignedUMAPAlignment",
    "evaluate_alignment",
    "flow_conservation_error",
    "compute_cosine_distance",
    "NeurIPSDataLoader",
    "ArXivDataLoader",
    "create_sample_neurips_data",
    "plot_temporal_evolution_2d",
    "plot_alignment_metrics_comparison",
    "plot_transport_couplings",
    "create_comprehensive_comparison_report",
]
