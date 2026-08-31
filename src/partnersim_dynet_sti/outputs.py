"""Script for generating output tables from multi-replicate STI simulations.

We are currently not using all the outputs from this script but they are
useful for exploratory analysis of the simulation results. There 
is a seperate script for generating plots which is in partnersim-dynet. 

``collate_demographic_summaries(sti_results)``
    Stack demographic_summary.csv files from all disease replicates into
    one DataFrame tagged with network_seed and disease_seed.

``summarise_by_group(collated_df, group_cols)``
    Aggregate across replicates: mean infection rate and reinfection rate
    per demographic group, with 95% CI.

``infection_rate_table(collated_df)``
    Convenience wrapper: returns a table broken down by
    sex x orientation x age_group, with separate columns for overall
    infection rate, infection rate among concurrency-allowed agents, and
    infection rate among agents who were ever actually concurrent.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from partnersim_dynet_sti.runner import STIRunResult


# Constant for demographic summary columns that are always present in the output CSV

_DEMO_COLS = ["sex", "orientation", "age_group"]
_ALL_GROUP_COLS = [*_DEMO_COLS, "concurrency_allowed", "ever_concurrent"]


# Functions for loading and aggregating demographic_summary.csv and disease_summary.csv

def collate_demographic_summaries(
    sti_results: list[STIRunResult],
) -> pd.DataFrame:
    """Load and stack demographic_summary.csv from every disease replicate.

    Parameters:

    sti_results : list[STIRunResult]
        Returned by ``run_sti_on_result``.

    Returns:

    DataFrame with all per-agent rows from all replicates, plus two
    additional columns:
        network_seed : int
        disease_seed : int

    Agents who never appeared in the partnership data are included with
    times_infected=0 and ever_infected=False, so denominators are correct
    when computing infection rates.
    """
    frames: list[pd.DataFrame] = []
    for r in sti_results:
        path = os.path.join(r.output_dir, "demographic_summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"demographic_summary.csv not found in {r.output_dir}. "
                "Make sure the model ran successfully."
            )
        df = pd.read_csv(path)
        df["network_seed"] = r.network_seed
        df["disease_seed"] = r.disease_seed
        frames.append(df)

    if not frames:
        raise ValueError("No STIRunResult objects provided.")

    return pd.concat(frames, ignore_index=True)

# Summarise infection and reinfection rates across replicates, stratified by demographic group
def summarise_by_group(
    collated: pd.DataFrame,
    group_cols: list[str] | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate infection and reinfection rates across disease replicates.

    For each demographic group, computes per-replicate infection rate then
    reports mean, SD, and CI across replicates.

    Parameters:

    collated : DataFrame
        Output of ``collate_demographic_summaries``.
    group_cols : list of str, optional
        Columns to group by.  Defaults to
        ``["sex", "orientation", "age_group"]``.
        Add ``"concurrency_allowed"`` or ``"ever_concurrent"`` if needed
    ci_level : float
        Confidence level for the CI (default 0.95).

    Returns:
    DataFrame with columns:
        <group_cols>,
        n_agents_mean,          mean number of agents in this group per replicate
        infection_rate_mean,    mean proportion ever infected
        infection_rate_sd,
        infection_rate_ci_lower,
        infection_rate_ci_upper,
        reinfection_rate_mean,  proportion of ever-infected who were reinfected
        reinfection_rate_sd,
        reinfection_rate_ci_lower,
        reinfection_rate_ci_upper,
        n_replicates
    """
    if group_cols is None:
        group_cols = _DEMO_COLS

    # Per-replicate rates is calculated by grouping by disease_seed + group_cols, then applying _replicate_rates
    per_rep = (
        collated
        .groupby(["disease_seed", *group_cols], observed=True)
        .apply(_replicate_rates, include_groups=False)
        .reset_index()
    )

    # Aggregate across replicates to get mean, SD, and CI for each group
    # The groups are defined by the group_cols, and we also keep track of how many replicates contributed to each group
    agg_cols = ["n_agents", "infection_rate", "reinfection_rate"]
    rows = []
    for group_vals, grp in per_rep.groupby(group_cols, observed=True):
        group_vals = (group_vals,) if not isinstance(group_vals, tuple) else group_vals
        row: dict = dict(zip(group_cols, group_vals, strict=True))
        n_reps = len(grp)
        row["n_replicates"] = n_reps
        for col in agg_cols:
            vals = grp[col].dropna().values
            row[f"{col}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{col}_sd"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if len(vals) > 1:
                lo, hi = _ci(vals, ci_level)
                row[f"{col}_ci_lower"] = lo
                row[f"{col}_ci_upper"] = hi
            else:
                row[f"{col}_ci_lower"] = np.nan
                row[f"{col}_ci_upper"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)

# Infection rate table by sex x orientation x age group
def infection_rate_table(collated: pd.DataFrame) -> pd.DataFrame:
    """Infection rates by sex x orientation x age group.

    Returns a table with one row per (sex, orientation, age_group)
    combination and columns for:

    - overall infection rate with 95% CI (all agents in that group)
    - infection rate for agents who were ever actually concurrent

    All rates are means across disease replicates.

    Parameters
    ----------
    collated : DataFrame
        Output of ``collate_demographic_summaries``.

    Returns
    -------
    DataFrame sorted by sex, orientation, age_group.
    """
    age_order = ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

    overall = summarise_by_group(collated, _DEMO_COLS)[
        [*_DEMO_COLS, "n_agents_mean", "infection_rate_mean",
         "infection_rate_ci_lower", "infection_rate_ci_upper",
         "reinfection_rate_mean", "n_replicates"]
    ].rename(columns={
        "infection_rate_mean":      "infection_rate",
        "infection_rate_ci_lower":  "infection_rate_ci_lo",
        "infection_rate_ci_upper":  "infection_rate_ci_hi",
        "reinfection_rate_mean":    "reinfection_rate",
    })

    # Proportion of agents in each group who were ever concurrent.
    # Computed directly from the collated demographic data — one value
    # per (disease_seed, group), then averaged across replicates.
    # Only recording concurrent mean, not CI. 
    conc_prop_rows = []
    for (d_seed, *group_vals), grp in collated.groupby(
        ["disease_seed", *_DEMO_COLS], observed=True
    ):
        n = len(grp)
        n_conc = grp["ever_concurrent"].sum()
        conc_prop_rows.append({
            **dict(zip(_DEMO_COLS, group_vals)),
            "disease_seed": d_seed,
            "concurrent_proportion": n_conc / n if n else 0.0,
        })
    conc_prop_df = pd.DataFrame(conc_prop_rows)
    conc_prop_mean = (
        conc_prop_df.groupby(_DEMO_COLS, observed=True)["concurrent_proportion"]
        .mean()
        .reset_index()
        .rename(columns={"concurrent_proportion": "concurrent_proportion_mean"})
    )

    # Merge the overall infection rates with the mean concurrent proportions to get a final table
    table = overall.merge(conc_prop_mean, on=_DEMO_COLS, how="left")

    # Age group is categorical, so we can sort by the order we want rather than alphabetically
    table["age_group"] = pd.Categorical(table["age_group"], categories=age_order, ordered=True)
    table = table.sort_values(["sex", "orientation", "age_group"]).reset_index(drop=True)

    return table


# Load disease_summary.csv from every replicate and stack into one DataFrame
def load_disease_summaries(sti_results: list[STIRunResult]) -> pd.DataFrame:
    """Load and stack disease_summary.csv from every disease replicate.

    Parameters:

    sti_results : list[STIRunResult]
        Returned by ``run_sti_on_result`` or ``run_sti_single``.

    Returns:
    
    DataFrame with columns:
        t, S, I, ever_reinfected, new_reinfections, R0, Reff,
        disease_seed, network_seed
        R0 and Reff are computed from the infection log, not from the model output.
        We are not currently using R0 and Reff in the analysis, but they are useful for exploratory analysis of the simulation results.
    """
    frames = []
    for r in sti_results:
        path = os.path.join(r.output_dir, "disease_summary.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"disease_summary.csv not found in {r.output_dir}. "
                "Make sure the model ran successfully."
            )
        df = pd.read_csv(path)
        df["disease_seed"] = r.disease_seed
        df["network_seed"] = r.network_seed
        frames.append(df)

    if not frames:
        raise ValueError("No STIRunResult objects provided.")

    return pd.concat(frames, ignore_index=True)

# Definitions of helper functions for summarising infection and reinfection rates
def _replicate_rates(group_df: pd.DataFrame) -> pd.Series:
    """Compute rates for one (disease_seed, group) slice."""
    n = len(group_df)
    n_infected = group_df["ever_infected"].sum()
    n_reinfected = (group_df["reinfection_count"] > 0).sum()
    return pd.Series(
        {
            "n_agents":        n,
            "infection_rate":  n_infected / n if n else 0.0,
            # Reinfection rate = fraction of ever-infected who were reinfected
            "reinfection_rate": n_reinfected / n_infected if n_infected else 0.0,
        }
    )

# Calculate a two-sided confidence interval for the mean of a sample using the t-distribution.
def _ci(values: np.ndarray, level: float) -> tuple[float, float]:
    """Two-sided t-interval for the mean of `values`."""
    n = len(values)
    if n < 2:
        return (np.nan, np.nan)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + level) / 2, df=n - 1)
    m = np.mean(values)
    return (float(m - h), float(m + h))