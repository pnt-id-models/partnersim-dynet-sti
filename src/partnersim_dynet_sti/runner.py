"""Entry points for running the SIS disease model.

Core function:

``run_sti_on_result(net_result, sis_cfg, seeds, output_dir)``
    Run one or more disease replicates on an single partnersim-dynet
    model. This is to keep the partnership network fixed and run multiple
    disease replicates on it. Returns a list of STIRunResult objects, 
    one per replicate. 
    
    net_result is a partnersim-dynet RunResult object, returned by
    partnersim_dynet.run_single() or partnersim_dynet.run_replicates().
    
    sis_cfg is a SISConfig object, which contains the disease parameters.
    seeds is a list of integer seeds, one per disease replicate. Each seed
    is used to initialise the random number generator for that replicate.
    output_dir is the directory where the disease outputs will be written.


Directory layout:

base_output_dir/
  network_seed_<N>/
    network/
      partnerships.parquet
      agent_log.parquet
      metrics.parquet              (if run_network_metrics=True)
      plots/                       (if run_network_plots=True)
      diagnostics/                 (if run_network_diagnostics=True)
    disease_seed_<D>/
      disease_summary.csv
      infection_events.csv
      demographic_summary.csv      <- one row per agent, infection + concurrency info

This file will include more options to run the disease model from scratch, including 
generating the network and running multiple replicates in parallel. 

"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from partnersim_dynet import RunResult as NetRunResult

from partnersim_dynet.network import ActiveIntervals, prepare_partnerships

from partnersim_dynet_sti.config import SISConfig, STISimulationConfig
from partnersim_dynet_sti.model import STISISModel

logger = logging.getLogger(__name__)


@dataclass
class STIRunResult:
    """Summary of one disease model run.

    Attributes:
    network_seed : int
        Seed used to generate the underlying partnership network.
    disease_seed : int
        Seed used for this disease replicate.
    output_dir : str
        Directory where disease outputs were written.
    peak_I_proportion : float
        Maximum proportion of active agents infected at any timestep.
    total_ever_infected : int
        Number of agents infected at least once.
    total_ever_reinfected : int
        Number of agents infected more than once.
    total_reinfection_events : int
        Total count of reinfection events (summed across all agents).
    """

    # This should match the fields in the dataclass definition above.  If you add new fields, update this list.
    network_seed: int
    disease_seed: int
    output_dir: str
    peak_I_proportion: float
    total_ever_infected: int
    total_ever_reinfected: int
    total_reinfection_events: int

# Core function to run one or more disease replicates on an existing partnersim-dynet RunResult.
def run_sti_on_result(
    net_result: NetRunResult,
    sis_cfg: SISConfig,
    seeds: int | list[int],
    output_dir: str,
    *,
    output_format: str = "parquet",
    verbose: bool = False,
) -> list[STIRunResult]:
    """Run one or more disease replicates on an existing partnersim-dynet RunResult.

    The partnership network is loaded once from disk and shared across all
    disease replicates, so running multiple disease replicates only
    requires one partnerships.parquet file.

    Parameters
    ----------
    net_result : RunResult
        Returned by ``partnersim_dynet.run_single``.
    sis_cfg : SISConfig
        Disease model parameters (shared across all replicates).
    seeds : int or list of int
        A single seed or a list of seeds — one disease run is executed per
        seed.  Passing ``seeds=list(range(20))`` runs 20 independent
        disease replicates on the same network.
    output_dir : str
        Root directory.  Each replicate is written to
        ``output_dir/disease_seed_<D>/``.
    output_format : str
        "parquet" or "csv" for loading the network files.
    verbose : bool

    Returns
    -------
    list[STIRunResult]
        One result per seed, in the same order as ``seeds``.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    seed_list: list[int] = [seeds] if isinstance(seeds, int) else list(seeds)

    # Load network files 
    net_dir = net_result.output_dir
    partnerships_df = _load_df(net_dir, "partnerships", output_format)
    agent_log = _load_df(net_dir, "agent_log", output_format)
    total_timesteps = _infer_total_timesteps(agent_log, partnerships_df)

    if verbose:
        logger.info(
            "Loaded network: %d partnerships, %d agents, T=%d",
            len(partnerships_df), len(agent_log), total_timesteps,
        )
        logger.info("Running %d disease replicate(s)", len(seed_list))

    results: list[STIRunResult] = []
    for d_seed in seed_list:
        d_dir = os.path.join(output_dir, f"disease_seed_{d_seed}")
        result = _run_sti_core(
            partnerships_df=partnerships_df,
            agent_log=agent_log,
            total_timesteps=total_timesteps,
            sis_cfg=sis_cfg,
            seed=d_seed,
            network_seed=net_result.seed,
            output_dir=d_dir,
            verbose=verbose,
        )
        results.append(result)

    return results


# Reconstruct a NetRunResult from a directory written by a previous run.  
# This is useful when you ran network replicates in a previous session (or in a separate script) 
# and want to point ``run_sti_on_result`` at one of them without re-running the network.
def net_result_from_dir(output_dir: str, seed: int) -> "NetRunResult":
    """Reconstruct a NetRunResult from a directory written by a previous run.

    Use this when you ran network replicates in a previous session (or in a
    separate script) and want to point ``run_sti_on_result`` at one of them
    without re-running the network.

    Parameters:
    output_dir : str
        Path to the directory that contains ``partnerships.parquet`` (or
        ``.csv``) and ``agent_log.parquet``.  This is the directory that was
        passed as ``output_dir`` to ``partnersim_dynet.run_single``.
    seed : int
        The seed that was used to generate that network.  Used to populate
        ``NetRunResult.seed`` so downstream functions can tag outputs correctly.

    Returns:
    NetRunResult
        A dataclass — only ``seed`` and ``output_dir`` are
        populated.  ``n_agents`` and ``n_partnerships`` are inferred from
        the files, otherwise set to -1.

    Example:
    >>> net = net_result_from_dir("results/network_seed_1234567/network", seed=1234567)
    >>> sti_results = run_sti_on_result(net, sis_cfg=SISConfig(), seeds=list(range(100)),
    ...                                  output_dir="results/network_seed_1234567")
    """
    
    # Try to infer counts from the files which will be present in the network output directory.  
    # If the files are missing or unreadable, n_agents and n_partnerships will be -1.
    n_agents = -1
    n_partnerships = -1
    for ext in ("parquet", "csv"):
        al_path = os.path.join(output_dir, f"agent_log.{ext}")
        pn_path = os.path.join(output_dir, f"partnerships.{ext}")
        if os.path.exists(al_path):
            try:
                al = pd.read_parquet(al_path) if ext == "parquet" else pd.read_csv(al_path)
                n_agents = len(al)
            except Exception:
                pass
        if os.path.exists(pn_path):
            try:
                pn = pd.read_parquet(pn_path) if ext == "parquet" else pd.read_csv(pn_path)
                n_partnerships = len(pn)
            except Exception:
                pass
        if n_agents != -1:
            break
    
    # Check that the required files exist in the output directory.  If not, raise a FileNotFoundError with a helpful message.
    if not os.path.exists(output_dir):
        raise FileNotFoundError(
            f"Network output directory not found: {output_dir}\n"
            "Check that the network was generated and the path is correct."
        )
    for stem in ("partnerships", "agent_log"):
        if not any(
            os.path.exists(os.path.join(output_dir, f"{stem}.{ext}"))
            for ext in ("parquet", "csv")
        ):
            raise FileNotFoundError(
                f"Required file '{stem}.parquet' or '{stem}.csv' not found in {output_dir}."
            )

    # Return a NetRunResult with the inferred counts and the provided seed and output_dir.
    return NetRunResult(
        seed=seed,
        output_dir=output_dir,
        n_agents=n_agents,
        n_partnerships=n_partnerships,
        files_written=[],
    )

# Function to run one disease replicate on an existing partnersim-dynet RunResult.
def _scalar(value):
    """Extract a plain Python scalar from an agentpy reporter value.

    agentpy stores reported values as single-element pandas Series.
    Calling int() or float() on a Series is deprecated in pandas >= 2.x
    and will raise a TypeError in a future release.  This helper unwraps
    the Series first, falling back for values that are already
    plain Python scalars.
    """
    import pandas as pd
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


# Core function to run one disease replicate on an existing partnersim-dynet RunResult.
# The function definition matches the signature of run_sti_on_result, but it only runs one replicate and returns a single STIRunResult.
def _run_sti_core(
    partnerships_df: pd.DataFrame,
    agent_log: pd.DataFrame,
    total_timesteps: int,
    sis_cfg: SISConfig,
    seed: int,
    network_seed: int,
    output_dir: str,
    verbose: bool,
) -> STIRunResult:
    """Build graph structures and run the agentpy model."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the ActiveIntervals and partnerships structures from the loaded dataframes. These are used to initialise the STISISModel.
    active = ActiveIntervals.from_agent_log(agent_log, total_timesteps=total_timesteps)
    partnerships = prepare_partnerships(partnerships_df, total_timesteps=total_timesteps)

    # If verbose, log the disease parameters and initial conditions for this replicate.
    if verbose:
        logger.info(
            "Disease seed=%d  beta=%.4f  gamma=%.4f  I0=%.1f%%  start_step=%d",
            seed,
            sis_cfg.infection_prob,
            sis_cfg.recovery_prob,
            sis_cfg.initial_infected * 100,
            sis_cfg.infection_start_step,
        )

    # These parameters are passed to the STISISModel constructor. 
    # They include the infection probability, recovery probability, 
    # initial infected proportion, infection start step, 
    # maximum timesteps, random seed, and population size.
    parameters = {
        "infection_prob":       sis_cfg.infection_prob,
        "recovery_prob":        sis_cfg.recovery_prob,
        "initial_infected":     sis_cfg.initial_infected,
        "infection_start_step": sis_cfg.infection_start_step,
        "max_steps":            total_timesteps,
        "seed":                 seed,
        "population":           len(agent_log),
    }

    model = STISISModel(
        parameters=parameters,
        partnerships=partnerships,
        active=active,
        agent_log=agent_log,
        partnerships_df=partnerships_df,
        output_dir=output_dir,
    )
    results = model.run()

    reported = dict(results.reporters) if hasattr(results, "reporters") else {}

    return STIRunResult(
        network_seed=network_seed,
        disease_seed=seed,
        output_dir=output_dir,
        peak_I_proportion=float(_scalar(reported.get("peak_I_proportion", 0.0))),
        total_ever_infected=int(_scalar(reported.get("total_ever_infected", 0))),
        total_ever_reinfected=int(_scalar(reported.get("total_ever_reinfected", 0))),
        total_reinfection_events=int(_scalar(reported.get("total_reinfection_events", 0))),
    )

# Load a parquet or csv file, trying both extensions.  Raises FileNotFoundError if neither exists.
def _load_df(directory: str, stem: str, fmt: str) -> pd.DataFrame:
    """Load a parquet or csv file, trying both extensions."""
    for ext in ("parquet", "csv"):
        path = os.path.join(directory, f"{stem}.{ext}")
        if os.path.exists(path):
            return pd.read_parquet(path) if ext == "parquet" else pd.read_csv(path)
    raise FileNotFoundError(
        f"Could not find {stem}.parquet or {stem}.csv in {directory}"
    )

# Infer total_timesteps from the data when it is not stored explicitly.  
# This is a fallback if the network script is changed to not write total_timesteps to the metrics file.  
# It looks for the maximum of ExitTimestep, EntryTimestep, and EndTime across the agent_log and partnerships files.
def _infer_total_timesteps(agent_log: pd.DataFrame, partnerships_df: pd.DataFrame) -> int:
    """Infer total_timesteps from the data when it is not stored explicitly."""
    candidates: list[int] = []

    if "ExitTimestep" in agent_log.columns:
        exits = agent_log["ExitTimestep"].dropna()
        if not exits.empty:
            candidates.append(int(exits.max()))

    if "EntryTimestep" in agent_log.columns:
        entries = agent_log["EntryTimestep"].dropna()
        if not entries.empty:
            candidates.append(int(entries.max()))

    if "EndTime" in partnerships_df.columns:
        ends = partnerships_df["EndTime"].dropna()
        if not ends.empty:
            candidates.append(int(ends.max()))

    if not candidates:
        raise ValueError(
            "Cannot infer total_timesteps from agent_log or partnerships_df. "
            "Pass it explicitly or use run_sti_single with an STISimulationConfig."
        )

    return max(candidates)