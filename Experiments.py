
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

# ----------------------------------------------------------------
# A simple test to check that different seeds produce different outcomes
# ----------------------------------------------------------------
# def test_seed_variation(algorithm_class: Type, 
#                         algo_params: Dict[str, Any], 
#                         num_runs: int = 5, 
#                         base_seed: int = 0, 
#                         parallel: bool = False) -> bool:
#     """
#     Run several experiments and check that the seed signatures are unique.
    
#     Returns True if all seed signatures are unique.
#     """
#     experiments = run_experiments(algorithm_class, algo_params, num_runs, base_seed, parallel)
#     signatures = [sig for (_, sig) in experiments]
#     return len(signatures) == len(set(signatures))

def test_seed_variation(algorithm_class: Type, 
                        algo_params: Dict[str, Any], 
                        num_runs: int = 5, 
                        base_seed: int = 0) -> None:
    """
    Run experiments sequentially and in parallel, print their seed signatures,
    and check that (a) within each mode the signatures are unique and (b) both modes
    produce the same set of signatures.
    """
    # Run sequential experiments
    seq_experiments = run_experiments(algorithm_class, algo_params, num_runs, base_seed, parallel=False)
    seq_signatures = [sig for (_, sig) in seq_experiments]
    print("Sequential experiment seed signatures:", seq_signatures)
    
    # Run parallel experiments
    par_experiments = run_experiments(algorithm_class, algo_params, num_runs, base_seed, parallel=True)
    par_signatures = [sig for (_, sig) in par_experiments]
    print("Parallel experiment seed signatures:", par_signatures)
    
    # Test that each mode produces unique signatures.
    seq_unique = len(seq_signatures) == len(set(seq_signatures))
    par_unique = len(par_signatures) == len(set(par_signatures))
    print("Sequential seed uniqueness test:", seq_unique)
    print("Parallel seed uniqueness test:", par_unique)
    
    # Test that both modes produce the same set of signatures.
    same_signatures = set(seq_signatures) == set(par_signatures)
    print("Sequential and parallel seeds contain the same set:", same_signatures)

# ----------------------------------------------------------------
# Example usage when running this file directly
# ----------------------------------------------------------------
if __name__ == '__main__':
    # Base parameters that are common for all experiments.
    base_params = {
        'sol_length': 100,                         # Length of the solution
        'opt_weights': (1.0,),                     # Maximization problem
        'eval_limit': 1000,                        # Maximum fitness evaluations
        'attr_function': binary_attribute,
        'fitness_function': (OneMax_fitness, {'noise_intensity': 0}),
        'true_fitness_function': (OneMax_fitness, {'noise_intensity': 0}),
        'starting_solution': None,
        'target_stop': None,
        'gen_limit': 100                           # Generation limit
    }
    # Algorithm-specific parameters for MuPlusLamdaEA.
    algo_params = base_params.copy()
    algo_params.update({
        'mu': 10,
        'lam': 1,
        'mutate_function': (tools.mutFlipBit, {'indpb': 1/100})
    })
    
    # # Run experiments sequentially.
    # sequential_results = run_experiments(MuPlusLamdaEA, algo_params, num_runs=5, base_seed=0, parallel=False)
    # print("Sequential experiment seed signatures:", [sig for (_, sig) in sequential_results])
    
    # # Run experiments in parallel.
    # parallel_results = run_experiments(MuPlusLamdaEA, algo_params, num_runs=5, base_seed=0, parallel=True)
    # print("Parallel experiment seed signatures:", [sig for (_, sig) in parallel_results])
    
    # Test that parallel runs use distinct seeds.
    test_seed_variation(MuPlusLamdaEA, algo_params, num_runs=5, base_seed=0)

