from RunManager import *

# ----------------------------------------------------------------
# Example Usage
# ----------------------------------------------------------------

if __name__ == '__main__':
    prob_info = {
        'name': 'OneMax100Item',
        'maximise': True,
        'opt_global': 100 
    }
    base_params = {
        'sol_length': 100,                         # Length of the solution.
        'opt_weights': (1.0,),                     # Maximization problem.
        'eval_limit': 100000,                        # Maximum fitness evaluations.
        'attr_function': binary_attribute,
        'starting_solution': None,
        'target_stop': 100,
        'gen_limit': None
    }
    fitness_functions = [(OneMax_fitness, {})]
    noise_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    algorithm_classes = [MuPlusLamdaEA, PCEA, UMDA, CompactGA]
    # algorithm_classes = [CompactGA]
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
                                num_runs=100,
                                base_seed=0,
                                parallel=True)
    pd.set_option('display.max_columns', None)
    print(results_df.head(20))
    results_df.to_pickle('results.pkl')