"""lut_native: direct LUT training for memory-constrained KAN deployment."""

from .baselines import (
    chebyshev_basis,
    dequantize_lut,
    eval_chebyshev,
    fit_chebyshev_ls,
    quantize_lut_uint8_asym,
    sample_polynomial_to_lut,
)
from .core import LUTEdge, lut_forward_numpy
from .coverage import (
    KAN2CoverageReport,
    LayerCoverageReport,
    compute_kan2_coverage,
    coverage_report_to_dict,
)
from .kan2 import (
    LUTKAN2Layer,
    kan2_forward_numpy,
    polynomial_kan2_forward_numpy,
)
from .kan2_low_rank import LowRankResidualLUTKAN2Layer
from .kan2_residual import ResidualLUTKAN2Layer
from .poly_kan2 import PolyKAN2Layer, train_poly_kan2
from .poly_kan2_zscore import PolyKAN2Zscore, train_poly_kan2_zscore
from .metrics import (
    cell_liveness,
    effective_rank,
    high_freq_energy,
    paired_bootstrap_ci,
    singular_value_spectrum,
)
from .regularizers import (
    boundary_continuity_penalty,
    boundary_slope_penalty,
    combined_penalty,
    first_diff_penalty,
    second_diff_penalty,
)
from .resources import (
    OpCount,
    ResourceProfile,
    lut_memory_bytes,
    lut_ops_per_sample,
    multi_edge_kan_ops,
    multi_edge_lut_memory_bytes,
    polynomial_memory_bytes,
    polynomial_ops_per_sample,
    time_forward,
)
from .targets import TARGETS, Target, generate_data, generate_data_2d
from .training import TrainConfig, TrainResult, train_lut_edge
from .training_kan2 import KAN2TrainConfig, KAN2TrainResult, train_kan2
from .training_residual import (
    ResidualTrainConfig,
    ResidualTrainResult,
    train_residual_kan2,
)
from .training_low_rank import (
    LowRankTrainConfig,
    LowRankTrainResult,
    train_low_rank_kan2,
)

__version__ = "0.2.0"

__all__ = [
    "LUTEdge", "lut_forward_numpy",
    "KAN2CoverageReport", "LayerCoverageReport",
    "compute_kan2_coverage", "coverage_report_to_dict",
    "LUTKAN2Layer", "kan2_forward_numpy", "polynomial_kan2_forward_numpy",
    "ResidualLUTKAN2Layer", "LowRankResidualLUTKAN2Layer",
    "PolyKAN2Layer", "train_poly_kan2",
    "PolyKAN2Zscore", "train_poly_kan2_zscore",
    "TrainConfig", "TrainResult", "train_lut_edge",
    "KAN2TrainConfig", "KAN2TrainResult", "train_kan2",
    "ResidualTrainConfig", "ResidualTrainResult", "train_residual_kan2",
    "LowRankTrainConfig", "LowRankTrainResult", "train_low_rank_kan2",
    "TARGETS", "Target", "generate_data", "generate_data_2d",
    "first_diff_penalty", "second_diff_penalty",
    "boundary_continuity_penalty", "boundary_slope_penalty",
    "combined_penalty",
    "chebyshev_basis", "fit_chebyshev_ls", "eval_chebyshev",
    "sample_polynomial_to_lut", "quantize_lut_uint8_asym", "dequantize_lut",
    "high_freq_energy", "cell_liveness", "effective_rank",
    "singular_value_spectrum", "paired_bootstrap_ci",
    "OpCount", "ResourceProfile",
    "lut_memory_bytes", "multi_edge_lut_memory_bytes", "polynomial_memory_bytes",
    "lut_ops_per_sample", "polynomial_ops_per_sample", "multi_edge_kan_ops",
    "time_forward",
]
