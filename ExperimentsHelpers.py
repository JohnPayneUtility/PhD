import pandas as pd
import os
from RunManager import *

# ==============================
# Saving
# ==============================

def save_or_append_results(df: pd.DataFrame, filename: str = 'results.pkl') -> None:
    """
    """
    if os.path.exists(filename):
        # Load the existing DataFrame
        existing_df = pd.read_pickle(filename)
        # Concatenate the existing and new DataFrames
        full_df = pd.concat([existing_df, df], ignore_index=True)
        # Save the combined DataFrame
        full_df.to_pickle(filename)
    else:
        # If the file doesn't exist, simply save the new DataFrame
        df.to_pickle(filename)

# ==============================
# Basis Experiment Runs
# ==============================

def dynamic_pop_size_UMDA(n_items, noise):
    return int(20 * np.sqrt(n_items) * np.log(n_items))

def dynamic_pop_size_PCEA(n_items, noise):
    return int(10 * np.sqrt(n_items) * np.log(n_items))

def inverse_n_mut_rate(n_items, noise):
    return {'indpb': 1/n_items}

# ===============================

def AlgosVariable(prob_info, base_params, fitness_functions, noise_values, runs, eval_limits = None):
    algorithm_classes = [MuPlusLamdaEA, UMDA]
    extra_params_by_algo = {
        'MuPlusLamdaEA': [
            # {'mu': 1, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
            {'mu': 1, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': inverse_n_mut_rate},
        ],
        'UMDA': [
            {'pop_size': dynamic_pop_size_UMDA},
        ],
        'PCEA': [
            {'pop_size': dynamic_pop_size_PCEA},
        ]
    }
    results_df = run_experiment(prob_info,
                                algorithm_classes,
                                fitness_functions,
                                noise_values,
                                extra_params_by_algo,
                                base_params,
                                eval_limits,
                                num_runs=runs,
                                base_seed=0,
                                parallel=True)
    pd.set_option('display.max_columns', None)
    # print(results_df.head(20))
    save_or_append_results(results_df)
    print(f"Completed problem {prob_info['name']}")

    # ===============================

def TwoAlgosTuned(prob_info, base_params, fitness_functions, noise_values, runs):
    algorithm_classes = [MuPlusLamdaEA, UMDA]
    extra_params_by_algo = {
        'MuPlusLamdaEA': [
            {'mu': 1, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
        ],
        'UMDA': [
            {'pop_size': 35},
        ]
    }
    results_df = run_experiment(prob_info,
                                algorithm_classes,
                                fitness_functions,
                                noise_values,
                                extra_params_by_algo,
                                base_params,
                                num_runs=runs,
                                base_seed=0,
                                parallel=True)
    pd.set_option('display.max_columns', None)
    # print(results_df.head(20))
    save_or_append_results(results_df)
    print(f"Completed problem {prob_info['name']}")

def FiveAlgosTuned(prob_info, base_params, fitness_functions, noise_values, runs):
    algorithm_classes = [MuPlusLamdaEA, PCEA, UMDA, CompactGA]
    extra_params_by_algo = {
        'MuPlusLamdaEA': [
            {'mu': 1, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
            {'mu': 5, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
        ],
        'PCEA': [
            {'pop_size': 22},
        ],
        'UMDA': [
            {'pop_size': 35},
        ],
        'CompactGA': [
            {'pop_size': 22},
        ]
    }
    results_df = run_experiment(prob_info,
                                algorithm_classes,
                                fitness_functions,
                                noise_values,
                                extra_params_by_algo,
                                base_params,
                                num_runs=runs,
                                base_seed=0,
                                parallel=True)
    pd.set_option('display.max_columns', None)
    # print(results_df.head(20))
    save_or_append_results(results_df)
    print(f"Completed problem {prob_info['name']}")

def FourAlgos(prob_info, base_params, fitness_functions, noise_values, runs):
    algorithm_classes = [MuPlusLamdaEA, PCEA, UMDA]
    extra_params_by_algo = {
        'MuPlusLamdaEA': [
            {'mu': 1, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
            {'mu': 100, 'lam': 1, 'mutate_function': tools.mutFlipBit, 'mutate_params': {'indpb': 1/100}},
        ],
        'PCEA': [
            {'pop_size': 100},
        ],
        'UMDA': [
            {'pop_size': 100},
        ],
    }
    results_df = run_experiment(prob_info,
                                algorithm_classes,
                                fitness_functions,
                                noise_values,
                                extra_params_by_algo,
                                base_params,
                                num_runs=runs,
                                base_seed=0,
                                parallel=True)
    pd.set_option('display.max_columns', None)
    # print(results_df.head(20))
    save_or_append_results(results_df)
    print(f"Completed problem {prob_info['name']}")
