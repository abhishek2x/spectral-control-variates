"""
Spectral Reduced-Variance Option Pricing Testbed.
Empirical validation of the spectral criterion for reduced-model control variates.
"""

from .config import (
    SimulationConfig,
    BachelierConfig,
    VasicekConfig,
    HestonConfig,
)
from .payoffs import (
    BasePayoff,
    EuropeanBasketCall,
    EuropeanBasketPut,
    OutOfSubspaceSpread,
    RainbowWorstOf,
)
from .spectral import (
    SpectralDecomposition,
    SpectralEngine,
)
from .simulators import (
    BaseSDESolver,
    BachelierSimulator,
    VasicekSimulator,
    HestonSimulator,
)
from .control_variates import (
    ControlVariateResult,
    ControlVariateEngine,
)

__all__ = [
    "SimulationConfig",
    "BachelierConfig",
    "VasicekConfig",
    "HestonConfig",
    "BasePayoff",
    "EuropeanBasketCall",
    "EuropeanBasketPut",
    "OutOfSubspaceSpread",
    "RainbowWorstOf",
    "SpectralDecomposition",
    "SpectralEngine",
    "BaseSDESolver",
    "BachelierSimulator",
    "VasicekSimulator",
    "HestonSimulator",
    "ControlVariateResult",
    "ControlVariateEngine",
]
