"""SIS disease model running on partnersim-dynet dynamic partnership networks.
"""

# These are the active functions and classes that are used to run the STI model and generate outputs.
from partnersim_dynet_sti.config import SISConfig, STISimulationConfig
from partnersim_dynet_sti.runner import STIRunResult, run_sti_on_result, net_result_from_dir

__all__ = [
    "SISConfig",
    "STISimulationConfig",
    "STIRunResult",
    "run_sti_on_result",
    "net_result_from_dir",        
    
]
__version__ = "0.1.0"