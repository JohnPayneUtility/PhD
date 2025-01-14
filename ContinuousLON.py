import numpy as np
from scipy.optimize import basinhopping, minimize
import os
import pickle

def rastrigin_eval(individual, amplitude=5):
    # print(f"Evaluating individual: {individual}")
    # print(f"Types in individual: {[type(x) for x in individual]}")
    A = amplitude
    n = len(individual)
    fitness = A * n + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual),
    return fitness

n_dimensions = 2
# x0 = np.random.uniform(-5.12, 5.12, n_dimensions)
all_runs_local_optima = []
all_runs_edges = []
all_runs_fitness_values = []


# Track the best result and the number of non-improving iterations for basin hoping
# class StopAfterNoImprovement:
#     def __init__(self, max_no_improve=1000):
#         self.best_fitness = float('inf')
#         self.no_improve_counter = 0
#         self.max_no_improve = max_no_improve

#     def __call__(self, x, f, accept):
#         if f < self.best_fitness:
#             self.best_fitness = f
#             self.no_improve_counter = 0
#         else:
#             self.no_improve_counter += 1

#         # Stop condition: 1000 iterations without improvement
#         if self.no_improve_counter >= self.max_no_improve:
#             return True  # Stop the optimization
#         return False

class BasinHoppingCallback:
    def __init__(self, precision=1e-5, max_no_improve=1000):
        self.local_optima = []  # To track local optima
        self.fitness_values = []  # To track fitness values for the current run
        self.edges = []
        self.previous_optimum = None  # Track the last local optimum
        self.best_fitness = float('inf')  # To track the best fitness
        self.no_improve_counter = 0  # Counter for non-improving iterations
        self.max_no_improve = max_no_improve  # Max allowed non-improving iterations
        self.precision = precision  # Precision for storing unique local optima

    def add_local_optimum(self, x, f):
        # Add unique local optima based on precision
        rounded_x = tuple(np.round(x, decimals=int(-np.log10(self.precision))))
        if rounded_x not in self.local_optima:
            self.local_optima.append(rounded_x)
            self.fitness_values.append(f)
        # Add edge
        if self.previous_optimum is not None and rounded_x != self.previous_optimum:
            self.edges.append((self.previous_optimum, rounded_x))
        self.previous_optimum = rounded_x

    def __call__(self, x, f, accept):
        # Track local optima
        self.add_local_optimum(x, f)
        
        # Check for improvement
        if f < self.best_fitness:
            self.best_fitness = f
            self.no_improve_counter = 0  # Reset counter
        else:
            self.no_improve_counter += 1

        # Stop if no improvement for the given threshold
        if self.no_improve_counter >= self.max_no_improve:
            return True  # Signal to stop basin hopping
        return False  # Continue

minimizer_kwargs = {
    "method": "L-BFGS-B",  # Local optimization method
    "bounds": [(-5.12, 5.12)] * n_dimensions,  # Bounds for the variables
    "options": {
        "gtol": 1e-7,  # Gradient tolerance (controls stopping when changes are small)
        "maxiter": 15000,  # Maximum iterations for the local search
    },
}

# Run basin-hopping
results = []
for _ in range(10):  # Multiple runs
    x0 = np.random.uniform(-5.12, 5.12, n_dimensions)  # Random start
    callback = BasinHoppingCallback(max_no_improve=1000)
    result = basinhopping(
        rastrigin_eval,
        x0=np.random.uniform(-5.12, 5.12, n_dimensions),  # Random start
        minimizer_kwargs=minimizer_kwargs,
        niter=10000,  # Maximum basin-hopping iterations
        stepsize=0.474,
        callback=callback,
    )
    results.append(result)

    # Save the local optima and fitnesses for this run
    all_runs_local_optima.append(callback.local_optima)
    all_runs_fitness_values.append(callback.fitness_values)
    all_runs_edges.append(callback.edges)

# Extract local optima and filter duplicates
local_optima = set(tuple(np.round(res.x, 5)) for res in results)
# local_optima = set(tuple(res.x) for res in results)

# Print results
print(f"Number of unique local optima: {len(local_optima)}")
for opt in local_optima:
    print(f"Optimum: {opt}")

# print(all_runs_local_optima)
# print(all_runs_fitness_values)
print(len(all_runs_local_optima))
print(len(all_runs_local_optima[0]))

local_optima_data = [all_runs_local_optima, all_runs_fitness_values, all_runs_edges]

folder = 'data/rastriginN2A5'
filename = 'MonotonicSequenceBasinHopping'
save_filename = f'{filename}_LO.pkl'
save_path = os.path.join(folder, save_filename)

os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(local_optima_data, file)
    
print(f"Local optima saved to {save_path}")