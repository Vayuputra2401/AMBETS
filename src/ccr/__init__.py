"""
CCR — Clinical Concept Routing package.

Top-level exports for the most commonly used objects.

    from ccr import CCRBottleneckModule, CCRConfig, CCRTotalLoss
"""

from ccr.config.ccr_config   import CCRConfig
from ccr.modules.dispatcher  import CCRBottleneckModule
from ccr.losses.total        import CCRTotalLoss
from ccr.utils.metrics       import ConceptAlignmentScore, ExpertUtilizationTracker
from ccr.utils.weight_schedule import CurriculumWeightScheduler

__all__ = [
    "CCRConfig",
    "CCRBottleneckModule",
    "CCRTotalLoss",
    "ConceptAlignmentScore",
    "ExpertUtilizationTracker",
    "CurriculumWeightScheduler",
]
