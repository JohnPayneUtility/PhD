
import random
import numpy as np
import pandas as pd
import concurrent.futures
import itertools
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Type

from deap import tools

from Algorithms import *


# ==============================
# Function to Conduct Single Run
# ==============================
def algo_data_single(prob_info: Dict[str, Any], 
                          algorithm_class: Type, 
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
        "problem_name": prob_info['name'],
        "problem_type": prob_info['type'],
        "problem_goal": prob_info['goal'],
        "dimensions": prob_info['dimensions'],
        "opt_global": prob_info['opt_global'],
        'PID': prob_info['PID'],
        "fit_func": algo_params['fitness_function'][0].__name__,
        "noise": algo_params['fitness_function'][1]['noise_intensity'],
        "algo_class": algorithm_class.__name__,
        "algo_type": algo_instance.type,
        "algo_name": algo_instance.name,
        "n_gens": algo_instance.gens,
        "n_evals": algo_instance.evals,
        # add eval limit
        "stop_trigger": algo_instance.stop_trigger,
        "n_unique_sols": len(unique_sols),
        "unique_sols": unique_sols,
        "unique_fits": unique_fits,
        "final_fit": unique_fits[-1],
        "max_fit": max(unique_fits),
        "min_fit": min(unique_fits),
        "sol_iterations": sol_iterations,
        "sol_transitions": sol_transitions,
        "seed": seed,
        "seed_signature": seed_signature,
    }

# # ==============================
# Function to Conduct Multiple Runs (sequentially or in parallel)
# # ==============================
def algo_data_multi(prob_info: Dict[str, Any],
                    algorithm_class: Type, 
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
                futures.append(executor.submit(algo_data_single, prob_info, algorithm_class, algo_params, seed))
            for future in concurrent.futures.as_completed(futures):
                results_list.append(future.result())
    else:
        for i in range(num_runs):
            seed = base_seed + i
            results_list.append(algo_data_single(prob_info, algorithm_class, algo_params, seed))
    
    # Create a DataFrame from the list of dictionaries.
    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values(by='seed')
    return df_sorted

# ==============================
# Function conduct experiment - gather algo data for range of values
# ==============================

def run_experiment(prob_info: Dict[str, Any],
                   algorithm_classes: List[Type],
                   fitness_functions: List[Tuple[Callable, dict]],
                   noise_values: List[float],
                   extra_params_by_algo: Dict[str, List[Dict[str, Any]]],
                   base_params: Dict[str, Any],
                   eval_limits: Optional[List[int]] = None,
                   num_runs: int = 10,
                   base_seed: int = 0,
                   parallel: bool = False) -> pd.DataFrame:
    """
    Run experiments over all combinations of settings for different algorithms.
    """
    results_list = []

    # Pair noise values with eval limits if provided
    if eval_limits is not None:
        assert len(eval_limits) == len(noise_values), \
            "Length of eval_limits must match noise_values."
        noise_eval_pairs = list(zip(noise_values, eval_limits))
    else:
        noise_eval_pairs = [(noise, base_params['eval_limit']) for noise in noise_values]

    for algo_class in tqdm(algorithm_classes, desc="Algorithm classes"):
        extra_param_list = extra_params_by_algo.get(algo_class.__name__, [{}])
        combinations = list(itertools.product(fitness_functions, noise_eval_pairs, extra_param_list))

        for fitness_fn, (noise, eval_limit), extra_params in tqdm(combinations,
                                                                  desc=f"Running {algo_class.__name__} configs",
                                                                  leave=False):
            params = base_params.copy()
            params.update(extra_params)

            # Evaluate callable parameters dynamically based on noise
            callable_params_keys = {'pop_size', 'mu', 'lam', 'eval_limit', 'mutate_params'}

            for key in callable_params_keys:
                if key in params and callable(params[key]):
                    params[key] = params[key](params['sol_length'], noise)

            # Set the eval_limit dynamically
            params['eval_limit'] = eval_limit  # Already extracted from noise_eval_pairs

            # Update fitness function with noise
            fit_params = fitness_fn[1].copy()
            fit_params.update({'noise_intensity': noise})
            params['fitness_function'] = (fitness_fn[0], fit_params)

            true_fit_params = fitness_fn[1].copy()
            true_fit_params.update({'noise_intensity': 0})
            params['true_fitness_function'] = (fitness_fn[0], true_fit_params)
            
            # Run the experiments for this configuration.
            df = algo_data_multi(prob_info, algo_class, params, num_runs, base_seed, parallel)

            # Add each key-value from extra_params as a column.
            for key, value in extra_params.items():
                if callable(value) and hasattr(value, '__name__'):
                    df[key] = value.__name__
                else:
                    df[key] = repr(value)
            
            results_list.append(df)
    
    final_df = pd.concat(results_list, ignore_index=True)
    return final_df