# IMPORTS
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

from tqdm import trange

# Fitness and solution functions
def generate_zero_solution(length):
    return np.zeros(length, dtype=int)

def random_bit_flip(bit_list, n_flips=1):
    for _ in range(n_flips):
        # Select a random index from the list
        index_to_flip = random.randint(0, len(bit_list) - 1)
        # Perform the bit flip (0 to 1 or 1 to 0)
        bit_list[index_to_flip] = 1 - bit_list[index_to_flip]
    return bit_list

def OneMax_fitness(individual, noise_function=None, noise_intensity=1):
    """ Function calculates fitness for OneMax problem individual """
    if noise_function is not None:
        individual = noise_function(individual[:], noise_intensity)
    fitness = sum(individual)
    return (fitness,)

def eval_ind_kp(individual, items_dict, capacity, penalty=1):
    """ Function calculates fitness for knapsack problem individual """
    n_items = len(individual)
    weight = sum(items_dict[i][1] * individual[i] for i in range(n_items)) # Calc solution weight
    value = sum(items_dict[i][0] * individual[i] for i in range(n_items)) # Calc solution value
    
    # Check if over capacity and return reduced value
    if weight > capacity:
        if penalty == 1:
            value_with_penalty = value - (weight - capacity)
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

# Population recording
def record_population_state(data, population, toolbox, true_fitness_function, true_fitness_params):
    # Unpack the data list
    all_generations, best_solutions, best_fitnesses, true_fitnesses = data

    # Record the current population, the best solution, and the true fitness if applicable
    all_generations.append([ind[:] for ind in population])
    best_individual = max(population, key=lambda ind: ind.fitness.values)  # Find the individual with the best fitness
    best_solutions.append(toolbox.clone(best_individual))  # Clone the best individual for recording
    best_fitnesses.append(best_individual.fitness.values[0])  # Record the best fitness value
    
    # If a true fitness function is provided, calculate the true fitness of the best solution
    if true_fitness_function is not None:
        if true_fitness_params is None:
            true_fitness_params = {}
        true_fitness = true_fitness_function(best_individual, **true_fitness_params)
        true_fitnesses.append(true_fitness[0])
    else:
        true_fitnesses.append(best_individual.fitness.values[0])

# Algorithm functions
# EA
def EA(NGEN, popsize, tournsize, MUTPB, indpb, len_sol, fitness_function, fitness_params=None, starting_solution=None, true_fitness_function=None, true_fitness_params=None, n_elite=1):
    # Check if the fitness and individual creators have been defined; if not, define them
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)

    # Define the toolbox used for the evolutionary algorithm
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)  # Define an attribute as a random binary value
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len_sol)  # Create an individual with len_sol attributes
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Create a population of individuals
    
    # Register the evaluation function, allowing additional parameters to be passed
    if fitness_params is None:
        fitness_params = {}
    toolbox.register("evaluate", lambda ind: fitness_function(ind, **fitness_params))
    
    # register mutation and selection functions
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)  # Mutation function: flip a bit with a given probability
    toolbox.register("select", tools.selTournament, tournsize=tournsize)  # Selection function: tournament selection

    # Create an initial population of individuals
    population = toolbox.population(n=popsize)

    # If a starting solution is provided, set all individuals to that solution
    if starting_solution is not None:
        for ind in population:
            ind[:] = starting_solution[:]

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Initialise data to record every generation's population, best solutions, best fitness values, and true fitness values
    all_generations, best_solutions, best_fitnesses, true_fitnesses = ([] for _ in range(4))
    data = [all_generations, best_solutions, best_fitnesses, true_fitnesses]
    # Record initial population
    record_population_state(data, population, toolbox, true_fitness_function, true_fitness_params)

    # Evolutionary loop for each generation
    for gen in trange(NGEN, desc='Evolving EA Solutions'):
        # Select the offspring using tournament selection (cloning to avoid modifying the original individuals)
        n_offspring = popsize - n_elite
        offspring = [toolbox.clone(toolbox.select(population, 1)[0]) for _ in range(n_offspring)]

        # Apply mutation on the offspring with probability MUTPB
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)  # Mutate the individual
                del mutant.fitness.values  # Delete fitness to mark it as needing reevaluation

        # Evaluate the individuals with an invalid fitness (those that were mutated)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the best individuals from the current population to keep (elitism)
        elites = tools.selBest(population, n_elite)

        # Replace the weakest individuals in the population with the offspring
        for mutant in offspring:
            weakest_idx = min(range(len(population)), key=lambda idx: population[idx].fitness.values)  # Find the weakest individual
            population[weakest_idx] = mutant  # Replace it with the mutated individual

        # Add the elite individuals back to the population
        # population.extend(elites)

        # Create the new population by combining elites and offspring
        population = elites + offspring

        # Record current population
        record_population_state(data, population, toolbox, true_fitness_function, true_fitness_params)

    return all_generations, best_solutions, best_fitnesses, true_fitnesses

# UMDA
def umda_update_full(len_sol, population, pop_size, select_size, replace_size, toolbox):
    # select from population
    selected_population = tools.selBest(population, select_size)

    # Calculate marginal propabilities
    probabilities = np.mean(selected_population, axis=0)

    new_solutions = []
    for _ in range(pop_size):
        new_solution = np.random.rand(len_sol) < probabilities
        new_solution = creator.Individual(new_solution.astype(int).tolist())  # Create as DEAP Individual
        new_solutions.append(new_solution)
    
    return new_solutions

def UMDA(NGEN, popsize, selectsize, len_sol, fitness_function, fitness_params=None, starting_solution=None, true_fitness_function=None, true_fitness_params=None):
    # Check if the fitness and individual creators have been defined; if not, define them
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)

    # Define the toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register the evaluation function, allowing additional parameters to be passed
    if fitness_params is None:
        fitness_params = {}
    toolbox.register("evaluate", lambda ind: fitness_function(ind, **fitness_params))

    # Create an initial population
    population = toolbox.population(n=popsize)

    # Set starting solution if provided
    if starting_solution is not None:
        for ind in population:
            ind[:] = starting_solution[:]
    
    # print([ind[:] for ind in population[:5]])

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Initialise data to record every generation's population, best solutions, best fitness values, and true fitness values
    all_generations, best_solutions, best_fitnesses, true_fitnesses = ([] for _ in range(4))
    data = [all_generations, best_solutions, best_fitnesses, true_fitnesses]
    
    # Record initial population
    record_population_state(data, population, toolbox, true_fitness_function, true_fitness_params)

    # Evolutionary loop
    for gen in trange(NGEN, desc='Evolving UMDA solution'):
        population = umda_update_full(len_sol, population, popsize, selectsize, 0, toolbox)

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Record current population
        record_population_state(data, population, toolbox, true_fitness_function, true_fitness_params)

    return all_generations, best_solutions, best_fitnesses, true_fitnesses

# Data processing functions
def extract_trajectory_data(best_solutions, best_fitnesses):
    # Extract unique solutions and their corresponding fitness values
    unique_solutions = []
    unique_fitnesses = []
    solution_iterations = []
    seen_solutions = {}

    for solution, fitness in zip(best_solutions, best_fitnesses):
        # Convert solution to a tuple to make it hashable
        solution_tuple = tuple(solution)
        if solution_tuple not in seen_solutions:
            seen_solutions[solution_tuple] = 1
            unique_solutions.append(solution)
            unique_fitnesses.append(fitness)
        else:
            seen_solutions[solution_tuple] += 1

    # Create a list of iteration counts for each unique solution
    for solution in unique_solutions:
        solution_tuple = tuple(solution)
        solution_iterations.append(seen_solutions[solution_tuple])

    return unique_solutions, unique_fitnesses, solution_iterations

def extract_transitions(unique_solutions):
    # Extract transitions between solutions over generations
    transitions = []

    for i in range(1, len(unique_solutions)):
        prev_solution = tuple(unique_solutions[i - 1])
        current_solution = tuple(unique_solutions[i])
        transitions.append((prev_solution, current_solution))

    return transitions

# Multiple runs and data saving
def conduct_runs(num_runs, algorithm_function, param_dict):
    " Function conducts multiple algorithm runs "
    all_run_trajectories = []
    
    for run in trange(num_runs, desc='Running multiple evolutions'):
        # Run the given algorithm with the provided parameters
        all_generations, best_solutions, best_fitnesses, true_fitnesses = algorithm_function(**param_dict)
        
        # Extract trajectories and transitions
        unique_solutions, unique_fitnesses, solution_iterations = extract_trajectory_data(best_solutions, true_fitnesses)
        transitions = extract_transitions(unique_solutions)
        
        # Store the results for each run
        all_run_trajectories.append((unique_solutions, unique_fitnesses, solution_iterations, transitions))
    
    return all_run_trajectories

def save_data(data, problem_name, algo_name):
    folder_path = 'data/' +  problem_name
    file_name = algo_name + '.pkl'

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the file within the folder
    file_path = os.path.join(folder_path, file_name)

    # Write file
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def save_parameters(parameters, problem_name, algo_name):
    folder_path = 'data/' +  problem_name
    file_name = algo_name + '.txt'

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the file within the folder
    file_path = os.path.join(folder_path, file_name)

    # Write file
    with open(file_path, 'w') as file:
        json.dump(str(parameters), file)
    
def save_problem(problem_info, problem_name):
    folder_path = 'data/' +  problem_name
    file_name = 'info' + '.txt'

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Create a serializable version of dictionary
    serializable_info = {
        key: value if isinstance(value, (int, float, str, bool, list, dict)) else str(value)
        for key, value in problem_info.items()
    }

    # Save the file within the folder
    file_path = os.path.join(folder_path, file_name)

    # Write file
    with open(file_path, 'w') as file:
        json.dump(serializable_info, file)

# Define problem and parameters and conduct runs
# Problem information
problem_name = 'OneMax_10items'
n_items = 100
problem_info = {
    'number of items': n_items,
}

# Algorithm information
EA_params = {
    'NGEN': 1000, # Number of generations
    'popsize': 100, # Population size
    'tournsize': 30, # Tournament selection size
    'MUTPB': 1, # Mutation probability
    'indpb': 0.05, # per-gene mutation probability
    'len_sol': n_items, # solution length
    'fitness_function': OneMax_fitness, # algorithm objective function
    'fitness_params': {'noise_function': random_bit_flip, 'noise_intensity': 0}, # objective function parameters
    'starting_solution': generate_zero_solution(n_items), # Specified starting solution for all individuals
    'true_fitness_function': OneMax_fitness, # noise-less fitness function for performance evaluation
    'true_fitness_params': {}, # noise-less fitnes function parameters
    'n_elite': 10
}
UMDA_params = {
    'NGEN': 500, # Number of generations
    'popsize': 100, # Population size
    'selectsize': 50, # Size selected for distribution
    'len_sol': n_items, # solution length
    'fitness_function': OneMax_fitness, # algorithm objective function
    'fitness_params': {'noise_function': random_bit_flip, 'noise_intensity': 3}, # objective function parameters
    'starting_solution': None, # Specified starting solution for all individuals
    'true_fitness_function': OneMax_fitness, # noise-less fitness function for performance evaluation
    'true_fitness_params': {} # noise-less fitnes function parameters
}

# conduct runs
algo_name = 'EA_g500_p100_t30_Test'
data = conduct_runs(3, EA, EA_params)
save_data(data, problem_name, algo_name)
save_parameters(EA_params, problem_name, algo_name)
save_problem(problem_info, problem_name)

# algo_name = 'UMDA_g500_p100_s50_Test'
# data = conduct_runs(1, UMDA, UMDA_params)
# save_data(data, problem_name, algo_name)
# save_parameters(EA_params, problem_name, algo_name)
# save_problem(problem_info, problem_name)

print('all runs complete')