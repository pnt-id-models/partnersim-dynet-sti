"""SIS disease model on a dynamic partnership network.

Uses partnersim-dynet's ``build_graph_at`` / ``PartnershipArrays`` /
``ActiveIntervals`` directly.

The model is driven by agentpy's event loop:
    setup() -> update() x T -> end()

``update()`` rebuilds the contact graph at each timestep using
``build_graph_at``, attaches agents to the graph's node data, then
run transmission and recovery.

Outputs -

disease_summary.csv      per-timestep S, I, R0, Reff, reinfection counts (We dont use R0 and Reff in our analysis at this stage)
infection_events.csv     every transmission event (infector, receiver, timestep)
demographic_summary.csv  one row per agent: infection counts + demographics + concurrent-partnership status
"""

from __future__ import annotations

import os

import agentpy as ap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from partnersim_dynet.network import ActiveIntervals, PartnershipArrays, build_graph_at

from partnersim_dynet_sti.agent import STISIS_Agent

# Age group labels matching partnersim-dynet conventions
_AGE_GROUPS = ["16-24", "25-34", "35-44", "45-54", "55-64", "65-74"]

# We define age groups as 16-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75+ (75+ is the default for any agent >= 75)
def _age_group(age: int) -> str:
    """Map integer age to the standard age-group label."""
    if age < 25:  return "16-24"
    if age < 35:  return "25-34"
    if age < 45:  return "35-44"
    if age < 55:  return "45-54"
    if age < 65:  return "55-64"
    if age < 75:  return "65-74"
    return "75+"

# This function is used to compute the proportion of agents in each demographic group who were ever concurrent. 
# It is used in the demographic summary output which is then used for plots. 
def _compute_ever_concurrent(partnerships_df: pd.DataFrame) -> dict[int, bool]:
    """Return {agent_id: bool} — True if the agent ever held 2+ simultaneous partnerships.

    Sweeps over each agent's partnership intervals.
    NaN EndTime (still active at sim end) included in the overlap check.

    Note: this is distinct from ConcurrencyAllowed in the agent log.
    ConcurrencyAllowed is a model parameter flag; ever_concurrent is an
    observational outcome — an agent can be allowed but never actually concurrent.
    """
    
    # real partnerships are those with a valid PartnerAgent and StartTime.  
    # Ignore any rows with missing values in these columns.
    real = partnerships_df[
        partnerships_df["PartnerAgent"].notna() & partnerships_df["StartTime"].notna()
    ].copy()
    if real.empty:
        return {}

    # Sentinel for "still active at sim end" — must be larger than any real timestep.
    sentinel = int(real["EndTime"].dropna().max()) + 1 if real["EndTime"].notna().any() else 999999

    # Use numpy arrays for fast boolean masking and sorting. 
    # This is implemented for columnswhose names begin with an underscor. 
    agents_arr = real["Agent"].to_numpy(dtype=np.int64)
    starts_arr = real["StartTime"].to_numpy(dtype=np.int32)
    ends_arr   = real["EndTime"].fillna(sentinel).to_numpy(dtype=np.int32)

    # Result dict: agent_id -> True if ever concurrent (2+ simultaneous partnerships)
    result: dict[int, bool] = {}
    
    # For each agent, build a list of events: +1 at each partnership start, -1 at each end.
    # Sort the events by time, then sweep through them to count how many partnerships are active at each time. 
    # If the count ever reaches 2 or more, mark the agent as ever concurrent   
    for agent_id, grp in real.groupby("Agent"):
        mask = agents_arr == agent_id
        starts = starts_arr[mask]
        ends   = ends_arr[mask]

        # Build the events: +1 at start, -1 at end
        events = sorted(
            [(int(s), +1) for s in starts] + [(int(e), -1) for e in ends]
        )
        max_simul = cur = 0
        for _, delta in events:
            cur += delta
            if cur > max_simul:
                max_simul = cur
        result[int(agent_id)] = max_simul >= 2

    return result


class STISISModel(ap.Model):
    """SIS transmission model on a dynamic partnership network.

    Parameters (passed via agentpy parameters dict)
    ------------------------------------------------
    infection_prob       : float   per-edge, per-step S->I probability
    recovery_prob        : float   per-step I->S probability
    initial_infected     : float   proportion seeded as I at infection_start_step
    infection_start_step : int     timestep when infections are first introduced
    max_steps            : int     total simulation length (= total_timesteps)
    seed                 : int     RNG seed

    Parameters added from partnersim-dynet
    -----------------------------------------------
    partnerships     : PartnershipArrays  pre-built from prepare_partnerships()
    active           : ActiveIntervals    pre-built from ActiveIntervals.from_agent_log()
    agent_log        : pd.DataFrame       raw agent log from the generator
    partnerships_df  : pd.DataFrame       raw partnership DataFrame (for concurrent detection)
    output_dir       : str                where to write outputs
    """

    # Initialisation
    def __init__(
        self,
        parameters: dict,
        partnerships: PartnershipArrays,
        active: ActiveIntervals,
        agent_log: pd.DataFrame,
        partnerships_df: pd.DataFrame,
        output_dir: str,
    ) -> None:
        super().__init__(parameters)
        self._partnerships = partnerships
        self._active = active
        self._agent_log = agent_log
        self._partnerships_df = partnerships_df
        self.output_dir = output_dir

        # Per-step time-series lists (index = t - 1)
        # This is to track the number of susceptible and infected agents at each timestep.
        self.S_counts: list[int] = []
        self.I_counts: list[int] = []
        
        # Per-step time-series lists for reinfection metrics.
        # Reinfection is defined as an agent being infected more than once.
        self.ever_reinfected_counts: list[int] = []
        self.new_reinfections_per_step: list[int] = []
        
        # Internal variable to track the cumulative number of reinfections across all agents.
        self._prev_reinfection_sum: int = 0

        # All outbound transmission events from agents
        # This is flushed each timestep to model.infection_events and then written to infection_events.csv at the end.
        self.infection_events: list[dict] = []

        # Populated in setup(). This is a mapping from agent ID to the corresponding STISIS_Agent instance.
        self._id_to_agent: dict[int, STISIS_Agent] = {}

    # Setup for the model. This is called once at the beginning of the simulation.
    def setup(self) -> None:
        np.random.seed(self.p.seed)

        all_agent_ids = self._agent_log["Agent"].tolist()
        N = len(all_agent_ids)
        self.p.population = N

        # Index agent_log by Agent once for fast row lookups
        log_idx = self._agent_log.set_index("Agent")

        # Create the agentpy AgentList with STISIS_Agent instances. 
        # Each agent is initialised with its ID and demographic attributes from the agent log.
        self.agents = ap.AgentList(self, N, STISIS_Agent)
        for agent, aid in zip(self.agents, all_agent_ids):
            agent.id = int(aid)
            # Row lookup in the agent log to get demographic attributes for this agent.
            row = log_idx.loc[aid]
            agent.sex = row["Sex"]
            agent.orientation = row["Orientation"]
            agent.age_group = _age_group(int(row["EntryAge"]))
            agent.concurrency_allowed = bool(row["ConcurrencyAllowed"])
            agent.entry_timestep = int(row["EntryTimestep"])
            agent.exit_timestep = (
                None if pd.isna(row["ExitTimestep"]) else int(row["ExitTimestep"])
            )
            self._id_to_agent[agent.id] = agent

        # If infection_start_step is 1, seed infections immediately. O
        # therwise, seeding will occur in the first update() call at infection_start_step.
        if self.p.infection_start_step <= 1:
            self._seed_infections()

    # Infections are seeded by randomly selecting a fraction of currently-active agents and marking them as infected (I).
    def _seed_infections(self) -> None:
        """Mark a random fraction of currently-active agents as I."""
        active_ids = self._active.active_at(self.t if hasattr(self, "_t") else 1)
        active_agents = [
            self._id_to_agent[aid] for aid in active_ids if aid in self._id_to_agent
        ]
        if not active_agents:
            return
        n = max(1, int(self.p.initial_infected * len(active_agents)))
        chosen = np.random.choice(active_agents, size=n, replace=False)
        for agent in chosen:
            agent.condition = "I"
            agent.times_infected = 1

    # Update function called at each timestep. This is where the main simulation logic occurs.
    def update(self) -> None:
        
        # check if we are at the infection start step and seed infections if needed
        t = self.t
        if t == self.p.infection_start_step and t > 1:
            self._seed_infections()
            
        # Rebuild the contact graph at this timestep using the partnership data and active intervals.
        graph = build_graph_at(t, self._partnerships, self._active)

        # For active agents, attach them to the graph's node data. Inactive agents are detached from the graph.
        # Inactive agents include those who have exited the population or are not currently in any partnerships.
        active_ids = self._active.active_at(t)
        for agent in self.agents:
            if agent.id in active_ids:
                agent.network = graph
                graph.nodes[agent.id]["agent"] = agent
            else:
                agent.network = None

        # If we are at or past the infection start step, run transmission and recovery for all agents.
        if t >= self.p.infection_start_step:
            for agent in self.agents:
                agent.transmission()
            for agent in self.agents:
                agent.recovery()

        # After transmission and recovery, collect all infection events from agents and reset their logs.
        for agent in self.agents:
            self.infection_events.extend(agent.infection_log)
            agent.infection_log = []

        # Compartment counts (active agents only). This is used to track the number of susceptible and infected agents at each timestep.
        active_agents = [
            self._id_to_agent[aid] for aid in active_ids if aid in self._id_to_agent
        ]
        n_active = len(active_agents)
        n_S = sum(1 for a in active_agents if a.condition == "S")
        n_I = sum(1 for a in active_agents if a.condition == "I")
        self.S_counts.append(n_S)
        self.I_counts.append(n_I)

        # We also track the proportion of susceptible and infected agents at each timestep, which is recorded in the model's output.
        self.record("S", n_S / n_active if n_active else 0.0)
        self.record("I", n_I / n_active if n_active else 0.0)
        self.record("n_active", n_active)

        # Reinfection - This is defined as an agent being infected more than once. 
        # We track the number of agents who have ever been reinfected and the number of new reinfections that occur at each timestep.
        # This is used to compute reinfection rates and is recorded in the model's output.
        ever_reinfected = sum(1 for a in self.agents if a.reinfection_count > 0)
        self.ever_reinfected_counts.append(ever_reinfected)
        current_sum = sum(a.reinfection_count for a in self.agents)
        self.new_reinfections_per_step.append(current_sum - self._prev_reinfection_sum)
        self._prev_reinfection_sum = current_sum

        # If we have reached the maximum number of steps, stop the simulation. This is controlled by the max_steps parameter.
        if t >= self.p.max_steps:
            self.stop()

    # End function called at the end of the simulation. This is where we write outputs to files and report summary statistics.
    def end(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        # If there are any infection events recorded, write them to infection_events.csv. This includes the timestep, infector agent ID, infected agent ID, whether it was a reinfection, and the infected agent's times infected.
        if self.infection_events:
            pd.DataFrame(self.infection_events).to_csv(
                os.path.join(self.output_dir, "infection_events.csv"), index=False
            )

        # Per-step summary
        n = len(self.S_counts)
        # n_active at each step to convert counts to proportions
        n_active_series = [
            s + i for s, i in zip(self.S_counts, self.I_counts)
        ]
        pd.DataFrame(
            {
                "t":                range(1, n + 1),
                "S":                self.S_counts,
                "I":                self.I_counts,
                "n_active":         n_active_series,
                "S_prop":           [s / na if na else 0.0
                                     for s, na in zip(self.S_counts, n_active_series)],
                "I_prop":           [i / na if na else 0.0
                                     for i, na in zip(self.I_counts, n_active_series)],
                "ever_reinfected":  self.ever_reinfected_counts,
                "ever_reinfected_prop": [er / na if na else 0.0
                                         for er, na in zip(self.ever_reinfected_counts,
                                                           n_active_series)],
                "new_reinfections": self.new_reinfections_per_step,
            }
        ).to_csv(os.path.join(self.output_dir, "disease_summary.csv"), index=False)

        # Demographic summary is written at the end of the simulation. This includes one row per agent with their demographics, infection counts, and whether they were ever concurrent.
        self._write_demographic_summary()

        # Reporters that collect summary metrics for total counts and peak prevalence
        total_ever_infected = sum(1 for a in self.agents if a.times_infected > 0)
        total_reinfected = sum(1 for a in self.agents if a.reinfection_count > 0)
        peak_I = max(self.I_counts) if self.I_counts else 0
        N = self.p.population

        self.report("total_ever_infected", total_ever_infected)
        self.report("total_ever_reinfected", total_reinfected)
        self.report("peak_I_count", peak_I)
        self.report("peak_I_proportion", peak_I / N if N else 0.0)
        self.report("total_reinfection_events", sum(a.reinfection_count for a in self.agents))

        print(
            f"[STI] seed={self.p.seed}  "
            f"peak_I={peak_I / N:.1%}  "
            f"ever_infected={total_ever_infected}  "
            f"reinfected={total_reinfected}  "
        )

    # Demographic summary for analysis of the infection counts and concurrency status of each agent
    # Note: This is distinct from the concurrency_allowed parameter in the agent log. 
    def _write_demographic_summary(self) -> None:
        """Write demographic_summary.csv — one row per agent.

        Columns
        -------
        agent_id, sex, orientation, age_group,
        concurrency_allowed, ever_concurrent,
        times_infected, reinfection_count, ever_infected
        """
        ever_concurrent = _compute_ever_concurrent(self._partnerships_df)

        records = []
        for agent in self.agents:
            records.append(
                {
                    "agent_id":           agent.id,
                    "sex":                agent.sex,
                    "orientation":        agent.orientation,
                    "age_group":          agent.age_group,
                    "concurrency_allowed": agent.concurrency_allowed,
                    "ever_concurrent":    ever_concurrent.get(agent.id, False),
                    "times_infected":     agent.times_infected,
                    "reinfection_count":  agent.reinfection_count,
                    "ever_infected":      agent.times_infected > 0,
                }
            )

        pd.DataFrame(records).to_csv(
            os.path.join(self.output_dir, "demographic_summary.csv"), index=False
        )

    