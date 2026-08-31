"""Run multiple disease replicates on a selected partnership network.

We use this to keep the network fixed while varying the disease random seed, so that we can
compare the effects of different disease parameters on the same network. 
Usage:
    poetry run python examples/run_disease_on_selected.py
"""

from partnersim_dynet_sti import net_result_from_dir, run_sti_on_result, SISConfig
from partnersim_dynet_sti.outputs import (
    collate_demographic_summaries,
    infection_rate_table,
)

# Choose a single network replicate to run multiple disease replicates on. 
# The network must have been generated with the same parameters as the disease model, and the partnership data must be available in the output directory.
REPLICATE_DIR = ("partnersim-dynet/examples/output/sweep/full_sweep_15pcconcurrency_15000agents_18Aug2026_#11/replicate_380863079")
SEED = 380863079
N_DISEASE_REPLICATES = 100
INFECTION_START_STEP = 51   # must match SISConfig below

# Load the network result from the selected replicate and run multiple disease replicates on it.
net = net_result_from_dir(output_dir=REPLICATE_DIR, seed=SEED)

# STI results are returned as a list of STIRunResult objects, one for each disease replicate. 
# Each object contains the output directory, network seed, and disease seed for that replicate.
sti_results = run_sti_on_result(
    net_result=net,
    sis_cfg=SISConfig(
        infection_prob=0.20,
        recovery_prob=0.10,
        infection_start_step=INFECTION_START_STEP,
        initial_infected=0.10,
    ),
    seeds=list(range(N_DISEASE_REPLICATES)),
    output_dir=REPLICATE_DIR,
    verbose=True,
)

# Demographic summaries include the information needed to generate the infection rate and reinfection rate tables
# This table is saved to a CSV file in the output directory for further analysis. 
# One per-agent row is included for every agent in the network, even if they were never infected.
collated = collate_demographic_summaries(sti_results)
table = infection_rate_table(collated)
table.to_csv(f"{REPLICATE_DIR}/infection_rate_table.csv", index=False)


print(f"\nDone. Results in: {REPLICATE_DIR}")