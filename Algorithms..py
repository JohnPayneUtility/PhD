# IMPORTS
import random
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Any

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# ==============================
# Mutation Functions
# ==============================

def mutSwapBit(individual, indpb):
    if random.random() < indpb and len(individual) >= 2:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return (individual,)

def complementary_crossover(parent1, parent2):
    assert len(parent1) == len(parent2), "Parents must have the same length."
    
    # Create empty offspring as lists
    offspring1 = type(parent1)([])
    offspring2 = type(parent2)([])
    
    # Generate the offspring
    for x1, x2 in zip(parent1, parent2):
        a = random.randint(0, 1)  # Randomly choose 0 or 1 with equal probability
        offspring1.append(a * x1 + (1 - a) * x2)
        offspring2.append((1 - a) * x1 + a * x2)

    return offspring1, offspring2

# ==============================
# Helper Functions
# ==============================

def record_population_state(data, population, toolbox, true_fitness_function):
    """
    Record the current state of the population.
    """
    all_generations, best_solutions, best_fitnesses, true_fitnesses = data

    # Record a snapshot of the current population
    all_generations.append([ind[:] for ind in population])
    
    # Identify the best individual in the current population
    best_individual = tools.selBest(population, 1)[0]
    best_solutions.append(toolbox.clone(best_individual))
    best_fitnesses.append(best_individual.fitness.values[0])
    
    # If provided, record the true (noise-free) fitness
    if true_fitness_function is not None:
        true_fit = true_fitness_function[0](best_individual, **true_fitness_function[1])
        true_fitnesses.append(true_fit[0])
    else:
        true_fitnesses.append(best_individual.fitness.values[0])

# ==============================
# Base Algorithm Class
# ==============================

@dataclass
class OptimisationAlgorithm:
    sol_length: int
    opt_weights: Tuple[float, ...]
    gen_limit: Optional[int] = 10e6
    eval_limit: Optional[int] = None
    target_stop: Optional[float] = None
    attr_function: Optional[Callable] = None
    fitness_function: Optional[Tuple[Callable, dict]] = None
    starting_solution: Optional[List[Any]] = None
    true_fitness_function: Optional[Tuple[Callable, dict]] = None
    
    # Fields that are not passed as init parameters can be defined with default_factory
    all_generations: List[List[Any]] = field(default_factory=list)
    best_solutions: List[Any] = field(default_factory=list)
    best_fitnesses: List[float] = field(default_factory=list)
    true_fitnesses: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.data = [self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses]

        # Fitness and individual creators
        if not hasattr(creator, "CustomFitness"):
            creator.create("CustomFitness", base.Fitness, weights=self.opt_weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.CustomFitness)

        # Create the toolbox and register common functions
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self.attr_function)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.sol_length)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", lambda ind: self.fitness_function[0](ind, **self.fitness_function[1]))
    
    @abstractmethod
    def run(self):
        """
        Run the evolutionary algorithm.
        Subclasses must implement this method.
        """
        pass

    def record_state(self, population):
        """
        Record the current population state.
        """
        record_population_state(self.data, population, self.toolbox, self.true_fitness_function)
    
    # def get_name(self):
    #     return self.name

    def get_algo_info():
        pass

# ==============================
# Evolutionary Algorithm Subclasses
# ==============================

class MuPlusLamdaEA(OptimisationAlgorithm):
    def __init__(self, 
                 mu: int,
                 lam: int, 
                 mutate_function: Tuple[Callable, dict], 
                 **kwargs):
        """
        (μ+1) Evolutionary Algorithm.
        
        Parameters:
            mu (int): Population size.
            lam (int): Recombination size.
            mutate_function (tuple): A tuple (function, params) for mutation.
            **kwargs: Other parameters passed to the base EvolutionaryAlgorithm.
        """
        # Initialize common components via the base class
        super().__init__(**kwargs)
        self.gens = 0
        self.evals = 0
        self.mu = mu
        self.lam = lam
        self.name = f'({mu}+{lam})EA'

        # Register the mutation operator in the toolbox
        self.toolbox.register("mutate", lambda ind: mutate_function[0](ind, **mutate_function[1]))

        # Create the initial population of size mu
        self.population = self.toolbox.population(n=mu)

        # If a starting solution is provided, initialize all individuals with it
        if self.starting_solution is not None:
            for ind in self.population:
                ind[:] = self.starting_solution[:]

        # Evaluate the initial population
        for ind in self.population:
            ind.fitness.values = self.toolbox.evaluate(ind)
        self.evals += mu

        # Record the initial state
        self.record_state(self.population)

    def run(self):
        """
        Execute the (μ+1) EA.
        
        Returns:
            A tuple of (all_generations, best_solutions, best_fitnesses, true_fitnesses).
        """
        while (self.gen_limit is None or self.gens < self.gen_limit):
            # If a maximum evaluation count is specified, check if it is reached.
            if self.eval_limit is not None and self.evals >= self.eval_limit:
                break
            # If a maximum fitness is specified, check if it is reached.
            if self.target_stop is not None and self.true_fitnesses[-1] >= self.target_stop:
                break

            self.gens += 1
            
            for offspring in range(self.lam):
                parent = random.choice(self.population)

                # Clone and mutate offspring                
                offspring = self.toolbox.clone(parent)
                offspring, = self.toolbox.mutate(offspring)
                
                # Re-evaluate the offspring
                del offspring.fitness.values
                offspring.fitness.values = self.toolbox.evaluate(offspring)
                self.evals += 1

                self.population.append(offspring)

            # Create new population from best solutions
            self.population = tools.selBest(self.population, self.mu)
            
            # Record the current population state
            self.record_state(self.population)
        
        return self.all_generations, self.best_solutions, self.best_fitnesses, self.true_fitnesses


# ==============================
# Estimation of Distribution Algorithm Subclasses
# ==============================




# ==============================
# Example Usage
# ==============================

if __name__ == "__main__":

    def binary_attribute():
        return random.randint(0, 1)
    
    mutation_rate = 1 / 100  # for a 100-bit solution
    mutate_function = (tools.mutFlipBit, {'indpb': mutation_rate})
    # mutate_function = (mutSwapBit, {'indpb': mutation_rate})

    from FitnessFunctions import OneMax_fitness
    fitness_function = (OneMax_fitness, {'noise_intensity': 0})
    true_fitness_function = (OneMax_fitness, {'noise_intensity': 0})

    ss = np.zeros(100, dtype=int)


    base_params = {
        'sol_length': 100,              # Length of the solution
        'opt_weights': (1.0,),           # Maximization problem
        'eval_limit': 10e4,               # Maximum fitness evaluations
        'attr_function': binary_attribute,
        'fitness_function': fitness_function,
        'true_fitness_function': true_fitness_function,
        'starting_solution': None
    }

    algo = MuPlusLamdaEA(mu=10, lam=1, mutate_function=mutate_function, **base_params)

    all_gens, best_sols, best_fits, true_fits = algo.run()
    
    # (Optional) Print the final best fitness
    print("Algo name:", algo.name)
    print("Final best fitness:", true_fits[-1])
    # print("First best sol:", best_sols[1])
