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
from datetime import datetime

from tqdm import trange

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def test_random_seed():
    print([random.randint(0, 100) for _ in range(5)])

# Fitness and solution functions
def generate_zero_solution(length):
    return np.zeros(length, dtype=int)

# def random_bit_flip(bit_list, n_flips=1):
#     for _ in range(n_flips):
#         # Select a random index from the list
#         index_to_flip = random.randint(0, len(bit_list) - 1)
#         # Perform the bit flip (0 to 1 or 1 to 0)
#         bit_list[index_to_flip] = 1 - bit_list[index_to_flip]
#     return bit_list

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
            value_with_penalty = capacity - weight
            return (value_with_penalty,)
        else:
            return (0,)
    return (value,) # Not over capacity return value

def rastrigin_eval(individual, amplitude=10):
    # print(f"Evaluating individual: {individual}")
    # print(f"Types in individual: {[type(x) for x in individual]}")
    A = amplitude
    n = len(individual)
    fitness = A * n + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual),
    return fitness

# Population recording
def record_population_state(data, population, toolbox, true_fitness_function):
    # Unpack the data list
    all_generations, best_solutions, best_fitnesses, true_fitnesses = data

    # Record the current population, the best solution, and the true fitness if applicable
    all_generations.append([ind[:] for ind in population])
    best_individual = tools.selBest(population, 1)[0] 
    best_solutions.append(toolbox.clone(best_individual))  # Clone the best individual for recording
    best_fitnesses.append(best_individual.fitness.values[0])  # Record the best fitness value
    
    # If a true fitness function is provided, calculate the true fitness of the best solution
    if true_fitness_function is not None:
        true_fitness = true_fitness_function[0](best_individual, **true_fitness_function[1])
        true_fitnesses.append(true_fitness[0])
    else:
        true_fitnesses.append(best_individual.fitness.values[0])

def timestamp():
    now = datetime.now()
    datetime_stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    return datetime_stamp

def euclidean_distance(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(list1, list2)))

# Algorithm functions
# 1+1 HC
def HC(NGEN, len_sol, weights, attr_function=None, mutate_function=None, purturbation=True, fitness_function=None, starting_solution=None, true_fitness_function=None):
    # Fitness and individual creators
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)
    
    # Check essential functions provided
    if attr_function is None:
        raise ValueError("Attribute generation function must be provided")
    if mutate_function is None:
        raise ValueError("Mutation function must be provided")

    # Define toolbox
    toolbox = base.Toolbox()
    # toolbox.register("attribute", random.randint, 0, 1)
    # toolbox.register("attribute", *attr_function)
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: fitness_function[0](ind, **fitness_function[1]))
    # toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))
    
    # Create population (1 for 1+1 EA)
    popsize = 1
    population = toolbox.population(n=popsize)

    if all(isinstance(item, int) for item in population[0]):
        attr_type = int
    else: attr_type = float
    print(attr_type)

    if all(isinstance(item, int) for item in population[0]): # check int or float
        toolbox.register("mutate", lambda ind, flipped_indices: mutate_function[0](ind, exclude_indices=flipped_indices, **mutate_function[1]))
    else: toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))

    # If a starting solution is provided, set all individuals to that solution
    if starting_solution is not None:
        for ind in population:
            ind[:] = starting_solution[:]

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Initialise data recording
    all_generations, best_solutions, best_fitnesses, true_fitnesses = ([] for _ in range(4))
    data = [all_generations, best_solutions, best_fitnesses, true_fitnesses]

    # Record initial population
    # record_population_state(data, population, toolbox, true_fitness_function)
    if not all(isinstance(item, int) for item in population[0]):
        record_population_state(data, population, toolbox, true_fitness_function)
    flipped_indices = set()

    # Evolutionary loop for each generation
    for gen in trange(NGEN, desc='Evolving EA Solutions'):
        # Generate a single mutant from the current solution
        mutant = toolbox.clone(population[0])
        if attr_type == int: # check int or float
            mutant[:], flipped_indices = toolbox.mutate(mutant, flipped_indices)
        else: mutant, = toolbox.mutate(mutant)
        del mutant.fitness.values  # Delete fitness to mark it as needing reevaluation

        # Evaluate the mutant
        mutant.fitness.values = toolbox.evaluate(mutant)

        # Replace the current solution with the mutant if it is better
        if tools.selBest([population[0], mutant], 1)[0] is mutant:
            population[0] = mutant
            flipped_indices = set()
            if not all(isinstance(item, int) for item in mutant):
                record_population_state(data, population, toolbox, true_fitness_function)
        

        # Record current population

        # record_population_state(data, population, toolbox, true_fitness_function)

        # Escape if all possible mutations applied with no improvement
        if len(flipped_indices) == len_sol:
            record_population_state(data, population, toolbox, true_fitness_function)
            if purturbation:
                attempts = 0
                attempt_limit = 1000
                while attempts < attempt_limit:
                    attempts += 1
                    mutant = toolbox.clone(population[0])
                    if attr_type == int: # check int or float # check int or float
                        mutant[:], flipped_indices = random_bit_flip(mutant, n_flips=2, exclude_indices=None)
                    else: mutant, = tools.mutGaussian(mutant, mu=0, sigma=0.2, indpb=0.5)
                    del mutant.fitness.values
                    mutant.fitness.values = toolbox.evaluate(mutant)
                    if tools.selBest([population[0], mutant], 1)[0] is mutant:
                        population[0] = mutant
                        flipped_indices = set()
                        break
                else: break
            else: break

    return all_generations, best_solutions, best_fitnesses, true_fitnesses


# EA
def EA(NGEN, popsize, tournsize, len_sol, weights, attr_function=None, mutate_function=None, fitness_function=None, starting_solution=None, true_fitness_function=None, n_elite=1):
    # if seed is not None:
    #     random.seed(seed)
    
    # Check if the fitness and individual creators have been defined; if not, define them
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)
    
    # Check essential functions provided
    if attr_function is None:
        raise ValueError("Attribute generation function must be provided")
    if mutate_function is None:
        raise ValueError("Mutation function must be provided")

    # Define toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: fitness_function[0](ind, **fitness_function[1]))
    toolbox.register("select", tools.selTournament, tournsize=tournsize)  # Selection function: tournament selection
    toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))

    # Create population
    population = toolbox.population(n=popsize)

    # If a starting solution is provided, set all individuals to that solution
    if starting_solution is not None:
        for ind in population:
            ind[:] = starting_solution[:]

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        # print(fit)
        ind.fitness.values = fit

    # Initialise data recording
    all_generations, best_solutions, best_fitnesses, true_fitnesses = ([] for _ in range(4))
    data = [all_generations, best_solutions, best_fitnesses, true_fitnesses]

    # Record initial population
    record_population_state(data, population, toolbox, true_fitness_function)

    # Evolutionary loop for each generation
    for gen in trange(NGEN, desc='Evolving EA Solutions'):
        # Select the best individuals from the current population to keep (elitism)
        # print(f"Generation {gen}: Best Fitness Before Elitism: {max(ind.fitness.values[0] for ind in population)}")
        elites = [toolbox.clone(ind) for ind in tools.selBest(population, n_elite)]
        # print(f"Elites Fitness: {[ind.fitness.values[0] for ind in elites]}")

        # Select the offspring using tournament selection (cloning to avoid modifying the original individuals)
        n_offspring = popsize - n_elite
        offspring = [toolbox.clone(toolbox.select(population, 1)[0]) for _ in range(n_offspring)]

        # Apply mutation on the offspring with probability MUTPB
        for mutant in offspring:
            toolbox.mutate(mutant)  # Mutate the individual
            del mutant.fitness.values  # Delete fitness to mark it as needing reevaluation

        # Evaluate the individuals with an invalid fitness (those that were mutated)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Create the new population by combining elites and offspring
        population = elites + offspring

        # Record current population
        record_population_state(data, population, toolbox, true_fitness_function)

    return all_generations, best_solutions, best_fitnesses, true_fitnesses

# UMDA
def umda_update_full(len_sol, population, pop_size, select_size, toolbox):
    # Select from population
    selected_population = tools.selBest(population, select_size)
    
    # Determine the data type of the genes from the first individual in the population
    gene_type = type(population[0][0])
    
    # Calculate marginal probabilities for binary solutions (assumes binary values are either 0 or 1)
    if gene_type == int:
        probabilities = np.mean(selected_population, axis=0)

        new_solutions = []
        for _ in range(pop_size):
            new_solution = np.random.rand(len_sol) < probabilities
            new_solution = creator.Individual(new_solution.astype(int).tolist())  # Create as DEAP Individual
            new_solutions.append(new_solution)

    # For float-based solutions, calculate mean and standard deviation
    elif gene_type == float:
        selected_array = np.array(selected_population)
        means = np.mean(selected_array, axis=0)
        stds = np.std(selected_array, axis=0)

        new_solutions = []
        for _ in range(pop_size):
            new_solution = np.random.normal(means, stds, len_sol)
            new_solution = creator.Individual(new_solution.tolist())  # Create as DEAP Individual
            new_solutions.append(new_solution)

    else:
        raise ValueError("Unsupported gene type. Expected int or float.")
    
    return new_solutions

def UMDA(NGEN, popsize, selectsize, len_sol, weights, attr_function=None, mutate_function=None, fitness_function=None, starting_solution=None, true_fitness_function=None):
    # Check if the fitness and individual creators have been defined; if not, define them
    if not hasattr(creator, "CustomFitness"):
        creator.create("CustomFitness", base.Fitness, weights=weights)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.CustomFitness)
    
    # Check essential functions provided
    if attr_function is None:
        raise ValueError("Attribute generation function must be provided")

    # Define toolbox
    toolbox = base.Toolbox()
    toolbox.register("attribute", attr_function)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len_sol)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: fitness_function[0](ind, **fitness_function[1]))

    # Create an initial population
    population = toolbox.population(n=popsize)

    # Set starting solution if provided
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
    record_population_state(data, population, toolbox, true_fitness_function)

    # Evolutionary loop
    for gen in trange(NGEN, desc='Evolving UMDA solution'):
        population = umda_update_full(len_sol, population, popsize, selectsize, toolbox)

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Record current population
        record_population_state(data, population, toolbox, true_fitness_function)

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

def serialize_parameters(parameters):
    """
    Convert any functions or non-serializable objects in a dictionary to strings,
    and ensure compatibility with JSON.
    """
    def convert_value(value):
        if callable(value):  # Check if it's a function or callable object
            return value.__name__
        elif isinstance(value, (np.integer, np.floating)):  # Handle NumPy numbers
            return value.item()
        elif isinstance(value, np.ndarray):  # Handle NumPy arrays
            return value.tolist()  # Convert array to list
        elif isinstance(value, tuple):  # Handle tuples that may contain functions or arrays
            return tuple(convert_value(v) for v in value)
        elif isinstance(value, dict):  # Handle nested dictionaries
            return {k: convert_value(v) for k, v in value.items()}
        elif isinstance(value, list):  # Handle lists that may contain non-serializable types
            return [convert_value(v) for v in value]
        return value  # Leave JSON-serializable objects unchanged

    return {key: convert_value(value) for key, value in parameters.items()}

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
    file_name = algo_name + '.json'

    serializable_params = serialize_parameters(parameters)

    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save the file within the folder
    file_path = os.path.join(folder_path, file_name)

    # Write file
    with open(file_path, 'w') as file:
        json.dump(serializable_params, file, indent=4)
    
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

def get_exp_name(function, parameters, suffix=''):
    from datetime import datetime
    function_name = function.__name__

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")
    gen = parameters['NGEN']
    if 'popsize' in parameters:
        pop = parameters['popsize']
    else: pop = 'NA'

    exp_name = f"{function_name}{suffix}_g{gen}_p{pop}_{timestamp}"
    return exp_name


def run_exp(algo, parameters, n_runs, problem_name, problem_info, suffix=''):
    name = get_exp_name(algo, parameters, suffix)
    data = conduct_runs(n_runs, algo, parameters)
    save_data(data, problem_name, name)
    save_parameters(parameters, problem_name, name)
    save_problem(problem_info, problem_name)
    print('Experiment Complete')




# problem_name = 'rastriginN2A10'


# attr_function = (random.uniform, -5.12, 5.12) # attribute function for rastrigin
# attr_function = (random.randint, 0, 1) # binary attribute function

# mutate_function = (tools.mutGaussian, {'mu': 0, 'sigma': 0.1, 'indpb': 0.05})
# mutate_function = (tools.mutGaussian, {'mu': 0, 'sigma': 0.1, 'indpb': 0.5})
# mutate_function = (tools.mutFlipBit, {'indpb': 0.01})

# fitness_function = (OneMax_fitness, {'noise_function': random_bit_flip, 'noise_intensity': 50})
# fitness_function = (rastrigin_eval, {'amplitude':10})
# fitness_function_true = (OneMax_fitness, {})
# fitness_function = (eval_ind_kp, {'items_dict': items_dict, 'capacity': capacity, 'penalty': 1})



def run_algorithm(args):
    """Wrapper function to run a single algorithm instance."""
    algorithm_function, param_dict = args

    # Extract the unique run_index and use it to seed the random number generator
    # run_index = param_dict.pop("run_index", None)

    all_generations, best_solutions, best_fitnesses, true_fitnesses = algorithm_function(**param_dict)

    # Extract trajectories and transitions
    unique_solutions, unique_fitnesses, solution_iterations = extract_trajectory_data(best_solutions, true_fitnesses)
    transitions = extract_transitions(unique_solutions)
    
    return unique_solutions, unique_fitnesses, solution_iterations, transitions

def conduct_runs_parallel(num_runs, algorithm_function, param_dict):
    """Conducts multiple algorithm runs in parallel."""
    args = [(algorithm_function, param_dict) for _ in range(num_runs)]
    # args = [(algorithm_function, {**param_dict, "run_index": index}) for index in range(num_runs)]

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(run_algorithm, args))

    return results

def run_exp_parallel(algo, parameters, n_runs, problem_name, problem_info, suffix=''):
    name = get_exp_name(algo, parameters, suffix)
    data = conduct_runs_parallel(n_runs, algo, parameters)
    save_data(data, problem_name, name)
    save_parameters(parameters, problem_name, name)
    save_problem(problem_info, problem_name)
    print('Experiment Complete')


def binary_attribute():
    """Generate a binary attribute."""
    return random.randint(0, 1)

def Rastrigin_attribute():
    return random.uniform(-5.12, 5.12)

def get_base_HC(attr_function, mutate_function, fitness_function, fit_weights, n_items):
    ss = generate_zero_solution(n_items)
    HC_params = {
        'NGEN': 10000, # Number of generations
        'len_sol': n_items, # solution length
        'weights': fit_weights,
        'attr_function': attr_function,
        'mutate_function': mutate_function,
        'fitness_function': fitness_function, # algorithm objective function
        'starting_solution': ss, # Specified starting solution for all individuals
        'true_fitness_function': None, # noise-less fitness function for performance evaluation
    }
    return HC_params

def get_base_EA(attr_function, mutate_function, fitness_function, fit_weights, n_items):
    ss = generate_zero_solution(n_items)
    EA_params = {
    'NGEN': 1000, # Number of generations
    'popsize': 100, # Population size
    'tournsize': 10, # Tournament selection size
    'len_sol': n_items, # solution length
    'weights': fit_weights,
    'attr_function': attr_function,
    'mutate_function': mutate_function,
    'fitness_function': fitness_function, # algorithm objective function
    'starting_solution': ss, # Specified starting solution for all individuals
    'true_fitness_function': None, # noise-less fitness function for performance evaluation
    'n_elite': 10
    }
    return EA_params

def get_base_UMDA(attr_function, fitness_function, fit_weights, n_items):
    UMDA_params = {
    'NGEN': 1000, # Number of generations
    'popsize': 100, # Population size
    'selectsize': 50, # Size selected for distribution
    'len_sol': n_items, # solution length
    'weights': fit_weights,
    'attr_function': attr_function,
    'mutate_function': None,
    'fitness_function': fitness_function, # algorithm objective function
    'starting_solution': None, # Specified starting solution for all individuals
    'true_fitness_function': None, # noise-less fitness function for performance evaluation
    }
    return UMDA_params

problem_names = [
        # 'f1_l-d_kp_10_269',
        # 'f2_l-d_kp_20_878',
        'f3_l-d_kp_4_20',
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

if __name__ == "__main__":
    n_runs_HC = 120
    n_runs = 12

    # KP problems
    for problem_name in problem_names:
        # Load problem, fitness, operations
        n_items, capacity, optimal, values, weights, items_dict, problem_info = load_problem_KP(problem_name)
        fit_weights = (1.0,)
        fitness_function = (eval_ind_kp, {'items_dict': items_dict, 'capacity': capacity, 'penalty': 1})
        mutate_function = (random_bit_flip, {'n_flips': 1})

        HC_params = get_base_HC(binary_attribute, mutate_function, fitness_function, fit_weights, n_items)
        # run_exp(HC, HC_params, n_runs, problem_name, problem_info, suffix='') # single threaded
        run_exp_parallel(HC, HC_params, n_runs_HC, problem_name, problem_info, suffix='') # multithreaded

        UMDA_params = get_base_UMDA(binary_attribute, fitness_function, fit_weights, n_items)
        run_exp_parallel(UMDA, UMDA_params, n_runs, problem_name, problem_info, suffix='')
        
        EA_params = get_base_EA(binary_attribute, mutate_function, fitness_function, fit_weights, n_items)
        run_exp_parallel(EA, EA_params, n_runs, problem_name, problem_info, suffix='')
    
    # Rastrigin problem
    # problem_name = 'rastriginN2A10'
    # n_items = 2
    # mutate_function = (tools.mutGaussian, {'mu': 0, 'sigma': 0.1, 'indpb': 0.5})
    # fitness_function = (rastrigin_eval, {'amplitude':10})
    # fit_weights = (-1.0,)

    # HC_params = get_base_HC(Rastrigin_attribute, mutate_function, fitness_function, fit_weights, n_items)
    # run_exp_parallel(HC, HC_params, n_runs_HC, problem_name, problem_info, suffix='') # multithreaded

    # UMDA_params = get_base_UMDA(Rastrigin_attribute, fitness_function, fit_weights, n_items)
    # run_exp_parallel(UMDA, UMDA_params, n_runs, problem_name, problem_info, suffix='')
    
    # EA_params = get_base_EA(Rastrigin_attribute, mutate_function, fitness_function, fit_weights, n_items)
    # run_exp_parallel(EA, EA_params, n_runs, problem_name, problem_info, suffix='')




