"""SIS agent for the STI disease model.

Each agent has a condition (S or I), and two counters that
track their infection history across the entire simulation run.

Model notes:

- ``infection_prob`` and ``recovery_prob`` are read from ``model.p`` and
  are identical for every agent. We assume homogeneous risk with the 
  partnership network.
- Recovery is a per-step Bernoulli draw, equivalent to a
  infectious period with mean ``1/recovery_prob`` steps.
- ``_receive_infection`` is called on the infected agent so that 
  reinfection counters are always updated.
- ``infection_log`` records transmission events caused by this agent and is 
  cleared to the model's central event store at the end of each step 
"""

from __future__ import annotations

import agentpy as ap
import numpy as np


class STISIS_Agent(ap.Agent):
    """Agent class for SIS transmission model.

    States:
    
    S : Susceptible — can be infected from connected nodes that are I .
    I : Infectious  — can transmit to S neighbours, recovers stochastically.

    Per-agent counters:
    
    times_infected : int
        Cumulative number of times this agent has been infected
        (including the initial seed infection).
    reinfection_count : int
        Number of infections *after* the first one, i.e.
        ``max(0, times_infected - 1)``.

    Per-step outbound log (cleared each step to model.infection_events)
    infection_log : list[dict]
        Transmission events caused by this agent this timestep.
        Moved into ``model.infection_events`` at the end of each step.
    """

    # Setup is called once at the start of the simulation to initialise agent state and counters. 
    # All agents start in the S state with zero infection history.
    def setup(self) -> None:
        self.condition: str = "S"
        self.times_infected: int = 0
        self.reinfection_count: int = 0
        self.infection_log: list[dict] = []

    # Internal helper to update infection state and counters when this agent receives an infection
    def _receive_infection(self, infector: "STISIS_Agent") -> None:
        """Transition to I and update counters.
        
        Parameters:
        infector : STISIS_Agent
            The agent responsible for this transmission event. The
            log entry is appended to the infector's ``infection_log`` so that
            that all events caused by an agent accumulate in one place.
        """
        # Track whether this is a reinfection (i.e. the agent has been infected before)
        was_previously_infected = self.times_infected > 0

        # If the agent is already infected, we don't change their state or counters, but we still log the event for the infector.
        self.condition = "I"
        self.times_infected += 1
        if was_previously_infected:
            self.reinfection_count += 1

        # Infector logs the event in their infection_log, which will be cleared to the model's central event store at the end of the step.
        infector.infection_log.append(
            {
                "timestep": self.model.t,
                "infector_agent": infector.id,
                "infected_agent": self.id,
                "is_reinfection": was_previously_infected,
                "receiver_times_infected": self.times_infected,
            }
        )

    # Transmission step: Called on every I agent each step to attempt to infect S neighbours
    def transmission(self) -> None:
        """Attempt to infect each S neighbour with probability infection_prob."""
        if (
            self.condition != "I"
            or not hasattr(self, "network")
            or self.network is None
            or self.id not in self.network
        ):
            return

        # Infection probability is read from the model parameters, which are assumed to be homogeneous across all agents.
        p_infect = self.model.p.infection_prob

        # For each neighbour of this agent in the network, if the neighbour is susceptible (S), attempt to infect them with probability p_infect.
        for neighbour in self.network.neighbors(self.id):
            nb: STISIS_Agent | None = self.network.nodes[neighbour].get("agent")
            # If the neighbour is None (not an agent) or not susceptible (not S), skip to the next neighbour.
            if nb is None or nb.condition != "S":
                continue
            if np.random.rand() < p_infect:
                nb._receive_infection(infector=self)

    # Recovery is called on every I agent each step to attempt to recover to S with probability recovery_prob
    def recovery(self) -> None:
        """Recover to S with probability recovery_prob."""
        if self.condition == "I" and np.random.rand() < self.model.p.recovery_prob:
            self.condition = "S"
            
    # Step is called on every agent each step to perform the transmission and recovery processes. It is the main entry point for the agent's behavior in the simulation.
    def step(self) -> None:
        self.transmission()
        self.recovery()