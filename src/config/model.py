"""Model and MCMC constants."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

_REGIONS = (
    "North West", "North East", "Yorkshire and The Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East", "South West",
)

# Representative cities for regional weather pulling
_REGIONAL_CENTERS = MappingProxyType({
    "London": (51.5074, -0.1278),
    "Manchester": (53.4808, -2.2426),
    "Birmingham": (52.4862, -1.8904),
    "Leeds": (53.8008, -1.5491),
    "Newcastle": (54.9783, -1.6178),
    "Bristol": (51.4545, -2.5879),
    "Norwich": (52.6309, 1.2974),
    "Southampton": (50.9097, -1.4044),
    "Nottingham": (52.9548, -1.1581),
})


@dataclass(frozen=True)
class ModelConfig:
    """Production values by default; use ``for_mode(test_mode=True)`` for fixtures."""

    RANDOM_SEED: int = 42
    # The synthesis emits a fixed, auditable number of synthetic households per
    # covered MSOA. It does not silently subsample the NEED seed or Census
    # tables; those are used as the IPF source/marginals before this explicit
    # expansion.
    N_HH_SAMPLES_PER_MSOA: int = 100

    COLD_SNAP_DURATION_HOURS: int = 336  # 14 days
    ELECTRIC_BASELOAD_KWH: int = 2000
    GAS_PRESENCE_THRESHOLD_KWH: int = 500

    # GP emulator held-out R^2 acceptance threshold. Lower under test mode
    # because the tiny fixture (a few hundred rows) cannot reliably reach the
    # production bar - a deliberately-chosen, still-enforced bar, not a bypass.
    GP_ACCEPTANCE_R2: float = 0.99

    # Base temperature for heating degree day calculations
    BASE_TEMP_HDD: float = 15.5

    MCMC_SAMPLES: int = 2000
    MCMC_TUNE: int = 3000  # Extra warmup for mass-matrix adaptation in the rho funnel
    MCMC_CORES: int = 4
    MCMC_CHAINS: int = 4

    # Convergence gate applied after every NUTS run, before results are treated
    # as final. A small nonzero divergence tolerance avoids an overly brittle
    # gate on an otherwise well-fit model; r_hat is only meaningful with >=2
    # chains, so model_unified.py checks it only when MCMC_CHAINS >= 2.
    MCMC_MAX_RHAT: float = 1.01
    MCMC_MAX_DIVERGENCES: int = 10

    REGIONS: tuple[str, ...] = _REGIONS
    REGIONAL_CENTERS: Mapping[str, tuple[float, float]] = _REGIONAL_CENTERS

    @classmethod
    def for_mode(cls, test_mode: bool = False) -> ModelConfig:
        base = cls()
        if not test_mode:
            return base
        return replace(
            base,
            MCMC_SAMPLES=10,
            MCMC_TUNE=10,
            MCMC_CORES=1,
            MCMC_CHAINS=1,
            GP_ACCEPTANCE_R2=0.95,
        )
