from RunManager import *
import pandas as pd
import os
from ProblemScripts import load_problem_KP


# ==============================
# Helper Functions
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

# ==============================
# Experiment Settings
# ==============================
basis_experiment = FourAlgos
eval_limit = 10000
runs = 30
selected_problems = [
        'onemax',
        'knapsack',
        'rastrigin',
    ]
noise_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
noise_values = [0, 3, 6, 9]



# ==============================
# Problem Specific Settings and Sub Experiment
# ==============================
rastrigin_dims = [2, 3]
kp_problems = [
    ('f10_l-d_kp_20_879', 1025),
    ('f1_l-d_kp_10_269', 295),
    # ('f2_l-d_kp_20_878', 1024),
    # ('f3_l-d_kp_4_20', 35),
    # ('f4_l-d_kp_4_11', 23),
    # ('f5_l-d_kp_15_375', 481.0694),
    # ('f6_l-d_kp_10_60', 52),
    # ('f7_l-d_kp_7_50', 107),
    # ('f8_l-d_kp_23_10000', 9767),
    # ('f9_l-d_kp_5_80', 130),
    # ('knapPI_1_10000_1000_1', 563647),
    # ('knapPI_1_1000_1000_1', 54503),
    # ('knapPI_1_100_1000_1', 9147),
    # ('knapPI_1_2000_1000_1', 110625),
    # ('knapPI_1_200_1000_1', 11238),
    # ('knapPI_1_5000_1000_1', 276457),
    # ('knapPI_1_500_1000_1', 28857),
    # ('knapPI_2_10000_1000_1', 90204),
    # ('knapPI_2_1000_1000_1', 9052),
    # ('knapPI_2_100_1000_1', 1514),
    # ('knapPI_2_2000_1000_1', 18051),
    # ('knapPI_2_200_1000_1', 1634),
    # ('knapPI_2_5000_1000_1', 44356),
    # ('knapPI_2_500_1000_1', 4566),
    # ('knapPI_3_10000_1000_1', 146919),
    # ('knapPI_3_1000_1000_1', 14390),
    # ('knapPI_3_100_1000_1', 2397),
    # ('knapPI_3_2000_1000_1', 28919),
    # ('knapPI_3_200_1000_1', 2697),
    # ('knapPI_3_5000_1000_1', 72505),
    # ('knapPI_3_500_1000_1', 7117),
]

if __name__ == '__main__':
    # ---------- ONEMAX ----------
    if 'onemax' in selected_problems:
        prob_info = {
            'name': 'OneMax',
            'type': 'Discrete',
            'goal': 'Maximization',
            'dimensions': 100,
            'opt_global': 100,
            'PID': 'OneMax_100' 
        }
        base_params = {
            'sol_length': 100,                         # Length of the solution.
            'opt_weights': (1.0,),                     # Maximization problem.
            'eval_limit': eval_limit,                        # Maximum fitness evaluations.
            'attr_function': binary_attribute,
            'starting_solution': None,
            'target_stop': 100,
            'gen_limit': None
        }
        fitness_functions = [(OneMax_fitness, {})]
        basis_experiment(prob_info, base_params, fitness_functions, noise_values, runs)
    
    # ---------- KNAPSACK ----------
    if 'knapsack' in selected_problems:
        for (filename, opt) in kp_problems:
            n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(filename)
        
            prob_info = {
                # 'name': filename,
                'name': 'Knapsack',
                'type': 'Discrete',
                'goal': 'Maximization',
                'dimensions': n_items,
                'opt_global': optimal,
                'PID':  filename
            }
            base_params = {
                'sol_length': n_items,                         # Length of the solution.
                'opt_weights': (1.0,),                     # Maximization problem.
                'eval_limit': eval_limit,                        # Maximum fitness evaluations.
                'attr_function': binary_attribute,
                'starting_solution': None,
                'target_stop': optimal,
                'gen_limit': None
            }
            fitness_functions = [
                (eval_noisy_kp_v1, {'items_dict': items_dict, 'capacity': capacity}),
                (eval_noisy_kp_v2, {'items_dict': items_dict, 'capacity': capacity}),
                ]
            basis_experiment(prob_info, base_params, fitness_functions, noise_values, runs)
    # ---------- Rastrigin ----------
    if 'rastrigin' in selected_problems:
        for dim in rastrigin_dims:
            prob_info = {
                'name': 'Rastrigin',
                'type': 'Continuous',
                'goal': 'Minimization',
                'dimensions': dim,
                'opt_global': 0.0,
                'PID': f'Rastrigin_{dim}D'
            }
            base_params = {
                'sol_length': dim,                         # Length of the solution.
                'opt_weights': (-1.0,),                     # Maximization problem.
                'eval_limit': eval_limit,                        # Maximum fitness evaluations.
                'attr_function': Rastrigin_attribute,
                'starting_solution': None,
                'target_stop': 0.0,
                'gen_limit': None
            }
            fitness_functions = [
                (rastrigin_eval, {})
                ]
            basis_experiment(prob_info, base_params, fitness_functions, noise_values, runs)
        pass

