from Algorithms import *
import optuna
import pandas as pd

# ==============================
# Parameter Optimisation Example
# ==============================

def objective(trial):
    # Suggest a candidate value for μ (mu). For example, between 5 and 50.
    # mu = trial.suggest_int("mu", 5, 100)
    # # lam = trial.suggest_int("lam", 1, 10)
    lam = 1
    # k = trial.suggest_int("k", 1, 10)

    # Mutation operator: using DEAP's mutFlipBit with a high mutation probability for demonstration.
    mutation_rate = 1 / 100  # For a 100-bit solution.
    mutate_function = tools.mutFlipBit
    mutate_params = {'indpb': mutation_rate}

    # Define fitness functions (using the dummy OneMax_fitness here).
    fitness_function = (OneMax_fitness, {'noise_intensity': 0})
    true_fitness_function = (OneMax_fitness, {'noise_intensity': 0})

    # Set up base parameters for your algorithm.
    base_params = {
        'sol_length': 100,              # Length of the solution
        'opt_weights': (1.0,),           # Maximization problem
        'eval_limit': 1000,              # Limit evaluations for testing
        'attr_function': binary_attribute,
        'fitness_function': fitness_function,
        'true_fitness_function': true_fitness_function,
        'starting_solution': None,
        'target_stop': 100,             # No early stopping target
        'gen_limit': 10000                 # Limit generations for tuning speed
    }
    
    # Create an instance of your MuPlusLamdaEA algorithm with the candidate μ.
    # algo = MuPlusLamdaEA(mu=mu, lam=lam, mutate_function=mutate_function, mutate_params=mutate_params, **base_params)
    pop_size = trial.suggest_int("pop", 4, 100)
    algo = PCEA(pop_size, **base_params)
    
    # Run the algorithm.
    algo.run()
    _, _, _, true_fits = algo.get_classic_data()
    
    # Retrieve the final best fitness (for OneMax, maximum is 100).
    final_fitness = true_fits[-1]
    
    # Return the final best fitness; Optuna will try to maximize this value.
    return (final_fitness, algo.evals)

# SINGLE OBJECTIVE
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000)
# print("Best parameters: ", study.best_params)
# print("Best objective: ", study.best_value)

# MULTI OBJECTIVEctive, n_trials=1000)
# best_trials = study
# study = optuna.create_study(directions=["maximize", "minimize"])
# study.optimize(obje.best_trials
# for trial in best_trials:
#     print("Trial params:", trial.params)
#     print("Trial values:", trial.values)

# Range 4-100
# 1000 trials

# mu + 1 EA 5
# cGA 22
# UMDA 35
# PCEA 22