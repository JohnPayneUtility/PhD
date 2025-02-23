
import random
import numpy as np
import pandas as pd
import concurrent.futures
from typing import List, Tuple, Any, Dict, Type

from deap import tools

from Algorithms import *
from FitnessFunctions import *

def run_single_experiment(algorithm_class: Type, 
                          algo_params: Dict[str, Any], 
                          seed: int) -> Dict[str, Any]:
    """
    Run one instance of the algorithm after setting the random seeds.
    
    The algorithm is expected to set its own seed_signature attribute and
    update its internal state. After running, we retrieve the trajectory data
    using get_trajectory_data() and the seed signature from the algorithm instance.
    
    Returns:
        A dictionary containing:
            - unique_sols: Unique solutions from the run.
            - unique_fits: Unique fitness values.
            - sol_iterations: Iteration counts for each unique solution.
            - sol_transitions: Transitions between unique solutions.
            - seed_signature: The seed signature stored in the algorithm instance.
    """
    # Set the random seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    
    # Create and run the algorithm instance.
    algo_instance = algorithm_class(**algo_params)
    algo_instance.run()  # This updates the instance's internal data.
    
    # Retrieve derived data from the run.
    unique_sols, unique_fits, sol_iterations, sol_transitions = algo_instance.get_trajectory_data()
    seed_signature = algo_instance.seed_signature
    
    return {
        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        "n_unique_sols": len(unique_sols),
        "unique_sols": unique_sols,
        "unique_fits": unique_fits,
        "final_fit": unique_fits[-1],
        "max_fit": max(unique_fits),
        "min_fit": min(unique_fits),
        "sol_iterations": sol_iterations,
        "sol_transitions": sol_transitions,
        "seed_signature": seed_signature,
    }

# ----------------------------------------------------------------
# Function to run multiple experiments (sequentially or in parallel)
# ----------------------------------------------------------------
def run_experiments(algorithm_class: Type, 
                    algo_params: Dict[str, Any], 
                    num_runs: int, 
                    base_seed: int = 0, 
                    parallel: bool = False) -> pd.DataFrame:
    """
    Run the given algorithm multiple times and collect results in a DataFrame.
    
    For each run, a unique seed is used (base_seed + run index). The function
    collects the trajectory data (unique solutions, fitness values, iteration counts,
    transitions) along with the seed signature from each run.
    
    Parameters:
      algorithm_class: The algorithm class to run (e.g., MuPlusLamdaEA or CompactGA).
      algo_params: A dictionary of parameters for initializing the algorithm.
      num_runs: Number of independent runs.
      base_seed: The starting random seed; each run will use base_seed + run index.
      parallel: If True, runs are executed in parallel.
      
    Returns:
      A Pandas DataFrame where each row corresponds to a run.
    """
    results_list = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_runs):
                seed = base_seed + i
                futures.append(executor.submit(run_single_experiment, algorithm_class, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results_list.append(run_single_experiment(algorithm_class, algo_params, seed))
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(results_list)
    return df

# ----------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------

if __name__ == '__main__':

    problem_info = {
        'problem_name': 'OneMax100Item',
        'maximise': True,
        'opt_global': 100 
    }
    base_params = {
        'sol_length': 100,                         # Length of the solution.
        'opt_weights': (1.0,),                     # Maximization problem.
        'eval_limit': 1000,                        # Maximum fitness evaluations.
        'attr_function': binary_attribute,
        'fitness_function': (OneMax_fitness, {'noise_intensity': 0}),
        'true_fitness_function': (OneMax_fitness, {'noise_intensity': 0}),
        'starting_solution': None,
        'target_stop': None,
        'gen_limit': 1000                           # Generation limit.
    }
    algo_params = base_params.copy()
    algo_params.update({
        'mu': 10,
        'lam': 1,
        'mutate_function': (tools.mutFlipBit, {'indpb': 1/100})
    })
    
    # Run experiments (sequentially or in parallel).
    results_df = run_experiments(MuPlusLamdaEA, algo_params, num_runs=10, base_seed=0, parallel=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(results_df.head())
