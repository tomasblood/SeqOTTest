"""
Global SeqOT: Sequential Optimal Transport with Global Optimization
Based on Watanabe & Isobe (2024/2025) - arXiv:2412.03120
"""

from .sinkhorn import ForwardBackwardSinkhorn
from .alignment import GlobalSeqOTAlignment, ProcrustesAlignment, AlignedUMAPAlignment
from .metrics import evaluate_alignment, flow_conservation_error
from .utils import compute_cosine_distance

__version__ = "1.0.0"
__all__ = [
    "ForwardBackwardSinkhorn",
    "GlobalSeqOTAlignment",
    "ProcrustesAlignment",
    "AlignedUMAPAlignment",
    "evaluate_alignment",
    "flow_conservation_error",
    "compute_cosine_distance",
]
