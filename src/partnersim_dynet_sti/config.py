"""Configuration dataclasses for the SIS disease model.

``SISConfig`` holds all disease-model parameters.
``STISimulationConfig`` combines a ``PartnershipConfig`` (from partnersim-dynet)
with a ``SISConfig`` and controls replication and output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from partnersim_dynet.config import PartnershipConfig


@dataclass
class SISConfig:
    """Parameters for one SIS disease model run.

    Transmission:
    
    infection_prob : float
        Per-edge, per-step probability of transmission from an I node to
        an S neighbour. Probabilities are same for every agent as we assume
        homogeneous risk of infection within the network.
    recovery_prob : float
        Per-step probability that an I agent recovers (returns to S).
        Equivalent to a geometric infectious period with mean
        ``1 / recovery_prob`` steps.

    Infection seeding:

    initial_infected : float
        Proportion of the population seeded as I at ``infection_start_step``.
    infection_start_step : int
        Timestep at which infections are first introduced.  Must be
        >= 1 and <= total_timesteps.  Set to 1 to seed at the very
        first timestep.  Using a value > 1 lets the partnership network 
        to reach steady-state structure before disease is introduced. 
        Partnerships are actively forming and dissolving from the very first
        timestep, so the network structure is not representative of the
        long-term steady-state until a few timesteps have passed.
    """

    # Transmission parameters
    infection_prob: float = 0.20
    recovery_prob: float = 0.10
    initial_infected: float = 0.10
    infection_start_step: int = 51   # allowing partnership network to reach steady-state structure before disease is introduced

    # Post-init validation of parameter ranges
    def __post_init__(self) -> None:
        if not 0.0 < self.infection_prob <= 1.0:
            raise ValueError("infection_prob must be in (0, 1]")
        if not 0.0 < self.recovery_prob <= 1.0:
            raise ValueError("recovery_prob must be in (0, 1]")
        if not 0.0 < self.initial_infected <= 1.0:
            raise ValueError("initial_infected must be in (0, 1]")
        if self.infection_start_step < 1:
            raise ValueError("infection_start_step must be >= 1")

    # Expected number of steps an agent remains infectious must be computed from the recovery probabilit
    def mean_infectious_period(self) -> float:
        """Expected number of steps an agent remains infectious."""
        return 1.0 / self.recovery_prob

    # Basic reproduction number ignoring network structure (R0 = beta/gamma). 
    @property
    def basic_r0_no_network(self) -> float:
        """R0 ignoring network structure (beta/gamma).

        The network-aware R0 at each timestep is computed inside the
        model as (beta/gamma) * mean_degree(t).
        Currently not used in the model, but useful for future analysis when a real infection is 
        introduced into the network. The network-aware R0 at each timestep is computed inside 
        the model as (beta/gamma) * mean_degree(t).
        """
        return self.infection_prob / self.recovery_prob


@dataclass
class STISimulationConfig:
    """Top-level config for a combined network + disease experiment.

    Generates a partnership network (via partnersim-dynet) and then
    runs one or more SIS disease replicates on it.

    All seeds are derived deterministically from ``base_network_seed``
    and ``base_disease_seed``.  Change only these values to get a
    different but still fully reproducible batch.

    ``network_seeds()`` returns ``n_network_replicates`` seeds.
    ``disease_seeds()`` returns ``n_disease_replicates`` seeds.

    Each network replicate gets every disease replicate run on it,
    giving ``n_network_replicates × n_disease_replicates`` total runs.
    """

    # Source configs for the two stages of the experiment from partnersim-dynet and partnersim-dynet-sti
    partnership: PartnershipConfig = field(default_factory=PartnershipConfig)
    sis: SISConfig = field(default_factory=SISConfig)

    # Replication parameters
    n_network_replicates: int = 1
    n_disease_replicates: int = 1
    base_network_seed: int = 1000
    base_disease_seed: int = 2000

    # output format and parallelism
    output_format: str = "parquet"   # "parquet" or "csv"
    verbose: bool = False
    n_workers: int = 1

    # partnersim-dynet analysis flags 
    # These are passed through to run_single so you get the network
    # diagnostics/plots alongside the disease outputs.
    run_network_metrics: bool = False
    run_network_plots: bool = False
    run_network_diagnostics: bool = False

    # Checks on parameter ranges and consistency
    def __post_init__(self) -> None:
        if self.n_network_replicates <= 0:
            raise ValueError("n_network_replicates must be positive")
        if self.n_disease_replicates <= 0:
            raise ValueError("n_disease_replicates must be positive")
        if self.output_format not in ("parquet", "csv"):
            raise ValueError("output_format must be 'parquet' or 'csv'")
        if self.n_workers <= 0:
            raise ValueError("n_workers must be positive")
        # Check that infection_start_step is within the total_timesteps of the partnership network
        if self.sis.infection_start_step > self.partnership.total_timesteps:
            raise ValueError(
                f"infection_start_step ({self.sis.infection_start_step}) "
                f"exceeds total_timesteps ({self.partnership.total_timesteps})"
            )
    
    # Deterministic seed generation for network and disease replicates
    # This is particularly useful for reproducibility when running multiple replicates in parallel, 
    # as it avoids seed collisions and ensures that each replicate is independent.
    def network_seeds(self) -> list[int]:
        """Deterministically derive per-network-replicate seeds."""
        rng = np.random.default_rng(self.base_network_seed)
        return rng.integers(0, 2**31, size=self.n_network_replicates).tolist()

    def disease_seeds(self) -> list[int]:
        """Deterministically derive per-disease-replicate seeds."""
        rng = np.random.default_rng(self.base_disease_seed)
        return rng.integers(0, 2**31, size=self.n_disease_replicates).tolist()