import random
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from ProblemScripts import load_problem_KP

import os
import pickle
import json
from datetime import datetime
from LON_Utilities import convert_to_split_edges_format

from tqdm import trange

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def OneMax_fitness(individual, noise_function=None, noise_intensity=0):
    """ Function calculates fitness for OneMax problem individual """
    if noise_function is not None: # Provide noise function for noise applied to individual
        individual = noise_function(individual[:], noise_intensity)
        fitness = sum(individual)
    else: # standard noisy
        fitness = sum(individual) + random.gauss(0, noise_intensity)
    return (fitness,)

def eval_ind_kp(individual, items_dict, capacity, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def binary_attribute():
    """Generate a binary attribute."""
    return random.randint(0, 1)

def random_bit_flip(bit_list, n_flips=1, exclude_indices=None):
    # test_random_seed()
    # Ensure n_flips does not exceed the length of bit_list
    n_flips = min(n_flips, len(bit_list))
    
    flipped_indices = set()
    if exclude_indices:
        flipped_indices.update(exclude_indices)
    if len(flipped_indices) == len(bit_list):
            return bit_list, flipped_indices

    for _ in range(n_flips):
        # Select a unique random index to flip
        index_to_flip = random.randint(0, len(bit_list) - 1)
        
        while index_to_flip in flipped_indices:
            index_to_flip = random.randint(0, len(bit_list) - 1)
        
        bit_list[index_to_flip] = 1 - bit_list[index_to_flip] # bit flip
        
        # Record the flipped index
        flipped_indices.add(index_to_flip)
        if len(flipped_indices) == len(bit_list):
            return bit_list, flipped_indices
    
    return bit_list, flipped_indices

def BinaryLON(pert_attempts, len_sol, weights, attr_function=None, n_flips_mut=1, n_flips_pert=2, mutate_function=None, perturb_function=None, improv_method='best', fitness_function=None, starting_solution=None, true_fitness_function=None):
    """
    """
    # Fitness and individual creators
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)

    # Helper function to generate all combinations of bit flips
    def generate_bit_flip_combinations(ind, n_flips):
        import itertools
        indices = range(len(ind))
        combinations = itertools.combinations(indices, n_flips)
        mutants = []
        for combo in combinations:
            mutant = toolbox.clone(ind)
            for index in combo:
                mutant[index] = 1 - mutant[index]
            mutants.append(mutant)
        return mutants

    # Define toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: fitness_function[0](ind, **fitness_function[1]))
    toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))
    toolbox.register("perturb", lambda ind: perturb_function[0](ind, **perturb_function[1]))
    
    # Create and evaluate starting solution
    if starting_solution is not None:
        individual = creator.Individual(starting_solution)
    else:
        individual = creator.Individual([toolbox.attribute() for _ in range(len_sol)])
    individual.fitness.values = toolbox.evaluate(individual)

    # Initialise data recording
    local_optima = []
    fitness_values = []
    edges = {}

    # Run basin-hopping and optimisation
    mut_attampts = 15000
    pert_attempts = 10000
    pert_attempt = 0
    while pert_attempt < pert_attempts:
        # USE CHOSEN OPTIMISATION METHOD TO OBTAIN LOCAL OPTIMUM
        # Use local search with first improvement or best improvement
        if improv_method=='first':
            pass

        if improv_method=='best':
            # optimisation of n_flip size neighbourhood for local optima
            # Generate a population of all possible mutants
            mut_attempt = 0
            # while mut_attempt < mut_attampts:
            mutants = generate_bit_flip_combinations(individual, n_flips_mut)
            for mutant in mutants:
                del mutant.fitness.values  # Mark as needing reevaluation
                mutant.fitness.values = toolbox.evaluate(mutant)

            best_mutant = tools.selBest(mutants + [individual], 1)[0]
            
            if best_mutant is not individual:
                if len(local_optima) > 0:


                    if tuple(individual) not in local_optima:
                        local_optima.append(tuple(individual))
                        fitness_values.append(individual.fitness.values[0])

                    if tuple(best_mutant) not in local_optima:
                        local_optima.append(tuple(best_mutant))
                        fitness_values.append(best_mutant.fitness.values[0])


                    edges[(tuple(individual), tuple(best_mutant))] = edges.get((tuple(individual), tuple(best_mutant)), 0) + 1
                individual[:] = best_mutant
                local_optima.append(tuple(individual))
                del individual.fitness.values
                individual.fitness.values = toolbox.evaluate(individual)
                fitness_values.append(individual.fitness.values[0])
            
            # Perform perturbation
            pert_attempt += 1
            perturbed = toolbox.clone(individual)
            perturbed[:], _ = random_bit_flip(perturbed, n_flips=n_flips_pert, exclude_indices=None)
            del perturbed.fitness.values
            perturbed.fitness.values = toolbox.evaluate(perturbed)

            if tools.selBest([individual, perturbed], 1)[0] is perturbed:
                individual[:] = perturbed
                pert_attempt = 0

    edges_list = [(source, target, weight) for (source, target), weight in edges.items()]
    return local_optima, fitness_values, edges_list

problem_names = [
        'f1_l-d_kp_10_269',
        # 'f2_l-d_kp_20_878',
        # 'f3_l-d_kp_4_20',
        # 'f4_l-d_kp_4_11',
        # 'f5_l-d_kp_15_375',
        # 'f6_l-d_kp_10_60',
        # 'f7_l-d_kp_7_50',
        # 'f8_l-d_kp_23_10000',
        # 'f9_l-d_kp_5_80',
        # 'f10_l-d_kp_20_879',
        # 'knapPI_1_100_1000_1',
        # 'knapPI_2_100_1000_1',
        # 'knapPI_3_100_1000_1'
    ]
# Iterate through selected problems
for problem_name in problem_names:
    n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(problem_name)
    fit_weights = (1.0,)
    fitness_function = (eval_ind_kp, {'items_dict': items_dict, 'capacity': capacity, 'penalty': 1})

    # Conduct runs of LON generation
    aggregated_lon_data = {
        "local_optima": [],
        "fitness_values": [],
        "edges": {},
        }
    
    for i in trange(100):
        local_optima, fitness_values, edges_list = BinaryLON(1000, n_items, fit_weights, attr_function=binary_attribute, n_flips_mut=2, n_flips_pert=4, mutate_function=None, perturb_function=None, improv_method='best', fitness_function=fitness_function, starting_solution=None, true_fitness_function=None)
        # print(local_optima)
        # print(fitness_values)
        # print(edges_list)

        for opt, fitness in zip(local_optima, fitness_values):
            if opt not in aggregated_lon_data["local_optima"]:
                aggregated_lon_data["local_optima"].append(opt)
                aggregated_lon_data["fitness_values"].append(fitness)

        # Aggregate edges
        for (source, target, weight) in edges_list:
            edge = (source, target)
            if edge in aggregated_lon_data["edges"]:
                aggregated_lon_data["edges"][edge] += weight  # Increment edge weight
            else:
                aggregated_lon_data["edges"][edge] = weight  # Add new edge with its weight

    # Save Aggregated LON for problem
    aggregated_lon_data_SE = convert_to_split_edges_format(aggregated_lon_data)
    folder = 'data/' +  problem_name
    filename = 'MonotonicSequenceBasinHopping'
    save_filename = f'{filename}_LO.pkl'
    save_path = os.path.join(folder, save_filename)
    os.makedirs(folder, exist_ok=True)
    with open(save_path, 'wb') as file:
            pickle.dump(aggregated_lon_data_SE, file)
    print(f"Local optima saved to {save_path}")

    # Save compressed LON
    def compress_lon_aggregated(LON_data, accuracy=1e-5):
        """
        Compress a Landscape of Optima Network (LON) by combining local optima with
        fitness values within a given accuracy threshold, aggregating edges accordingly.

        Args:
            LON_data (dict): Dictionary containing LON data with keys:
                - "local_optima": List of unique local optima (each a tuple).
                - "fitness_values": List of fitness values (float or int) corresponding to the local optima.
                - "edges": Dictionary with keys as (source, target) tuples (each source/target a tuple),
                        and values as numeric edge weights.
            accuracy (float): Threshold for grouping local optima with close fitness values.

        Returns:
            dict: A new LON dictionary with the same format and types, but aggregated
                according to the provided accuracy.
        """

        # ----------------------------------------------------
        # 1. Group local optima based on fitness similarity
        # ----------------------------------------------------
        grouped_optima = []          # Representative local optima for each group
        grouped_fitness_values = []  # Representative fitness for each group
        membership = []              # For each original local optimum, which group does it belong to?

        for opt, fit in zip(LON_data["local_optima"], LON_data["fitness_values"]):
            assigned_group = None
            # Check if this fitness is close enough to a group representative's fitness
            for g_idx, g_fit in enumerate(grouped_fitness_values):
                if abs(fit - g_fit) <= accuracy:
                    assigned_group = g_idx
                    break

            # If not found in any group, create a new group
            if assigned_group is None:
                grouped_optima.append(opt)
                grouped_fitness_values.append(fit)
                assigned_group = len(grouped_optima) - 1

            membership.append(assigned_group)

        # ----------------------------------------------------
        # 2. Build a new edge dictionary based on the groups
        # ----------------------------------------------------
        # map each original local optimum to an index to easily find its group
        opt_to_index = {opt: i for i, opt in enumerate(LON_data["local_optima"])}

        new_edges = {}
        for (source, target), weight in LON_data["edges"].items():
            # Identify the groups of the source and target
            source_group = membership[opt_to_index[source]]
            target_group = membership[opt_to_index[target]]

            # The new source/target in the aggregated LON
            new_source = grouped_optima[source_group]
            new_target = grouped_optima[target_group]

            # Aggregate edge weights if the same group-pair already exists
            if (new_source, new_target) not in new_edges:
                new_edges[(new_source, new_target)] = weight
            else:
                new_edges[(new_source, new_target)] += weight

        # ----------------------------------------------------
        # 3. Construct the new, aggregated LON data structure
        # ----------------------------------------------------
        compressed_lon_data = {
            "local_optima": grouped_optima,
            "fitness_values": grouped_fitness_values,
            "edges": new_edges,
        }

        # ----------------------------------------------------
        # 4. Validate output to ensure it matches required format
        # ----------------------------------------------------
        assert isinstance(compressed_lon_data, dict), "Output must be a dictionary."
        assert "local_optima" in compressed_lon_data, "Output dictionary must contain 'local_optima'."
        assert "fitness_values" in compressed_lon_data, "Output dictionary must contain 'fitness_values'."
        assert "edges" in compressed_lon_data, "Output dictionary must contain 'edges'."
        assert all(isinstance(opt, tuple) for opt in compressed_lon_data["local_optima"]), \
            "All local optima in 'local_optima' must be tuples."
        assert all(isinstance(f, (float, int)) for f in compressed_lon_data["fitness_values"]), \
            "All values in 'fitness_values' must be numeric."
        assert all(
            isinstance(k, tuple) and len(k) == 2 
            and isinstance(k[0], tuple) and isinstance(k[1], tuple)
            for k in compressed_lon_data["edges"].keys()
        ), "All edge keys must be 2-tuples of solutions (which are tuples)."
        assert all(isinstance(v, (float, int)) for v in compressed_lon_data["edges"].values()), \
            "All edge weights must be numeric."

        return compressed_lon_data

    compressed_lon_data = compress_lon_aggregated(aggregated_lon_data, accuracy=1e-4)
    compressed_lon_data_SE = convert_to_split_edges_format(compressed_lon_data)
    save_filename = f'{filename}_CLO.pkl'
    save_path = os.path.join(folder, save_filename)
    os.makedirs(folder, exist_ok=True)
    with open(save_path, 'wb') as file:
            pickle.dump(compressed_lon_data_SE, file)
    print(f"Compressed Local optima saved to {save_path}")