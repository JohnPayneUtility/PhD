
import random
import numpy as np
import concurrent.futures
from typing import List, Tuple, Any, Dict, Type

from deap import tools

from Algorithms import *
from FitnessFunctions import *

def run_single_experiment(algorithm_class: Type, algo_params: Dict[str, Any], seed: int) -> Tuple[Any, int]:
    """
    Run one instance of the algorithm after setting the random seeds.
    
    Returns:
        A tuple of (algorithm_result, seed_signature) where seed_signature is a
        random number generated after running the algorithm (for testing).
    """
    # Set the random seeds for reproducibility in this run.
    random.seed(seed)
    np.random.seed(seed)
    
    # Create an instance of the algorithm.
    algo_instance = algorithm_class(**algo_params)
    # Run the algorithm.
    result = algo_instance.run()
    # Generate a "seed signature" by drawing a random number.
    seed_signature = random.randint(0, 10**6)
    return result, seed_signature

# ----------------------------------------------------------------
# Function to run multiple experiments (sequentially or in parallel)
# ----------------------------------------------------------------
def run_experiments(algorithm_class: Type, 
                    algo_params: Dict[str, Any], 
                    num_runs: int, 
                    base_seed: int = 0, 
                    parallel: bool = False) -> List[Tuple[Any, int]]:
    """
    Run the given algorithm multiple times and collect the results.
    
    Parameters:
      algorithm_class: The algorithm class to run (e.g., MuPlusLamdaEA).
      algo_params: A dictionary of parameters for initializing the algorithm.
      num_runs: Number of independent runs.
      base_seed: The starting random seed; each run will use base_seed + run index.
      parallel: If True, runs are executed in parallel.
      
    Returns:
      A list of tuples, each containing the algorithm's result and a seed signature.
    """
    results = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_runs):
                seed = base_seed + i
                futures.append(executor.submit(run_single_experiment, algorithm_class, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results.append(run_single_experiment(algorithm_class, algo_params, seed))
    return results



