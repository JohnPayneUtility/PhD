import numpy as np
from scipy.optimize import basinhopping, minimize
import os
import pickle

def rastrigin_eval(individual, amplitude=5):
    """

    """
    A = amplitude
    n = len(individual)
    fitness = A * n + sum((x ** 2 - A * np.cos(2 * np.pi * x)) for x in individual),
    return fitness

def birastrigin_eval(individual, d=1, s=None):
    """
    Fitness evaluation for the Birastrigin problem

    Args:
        individual (list or np.ndarray): The input vector representing an individual.
        d (float, optional): Parameter `d`, standardized to 1 unless specified otherwise.
        s (float, optional): Parameter `s`, if not provided, it is calculated as per the formula.

    Returns:
        tuple: A single-element tuple containing the fitness value.
    """
    # Define parameters
    mu1 = 2.5
    if s is None:
        s = 1 - (1 / (2 * np.sqrt(2) + 20 - 8.2))
    mu2 = -np.sqrt(mu1**2 - d / s)

    n = len(individual)

    # Compute the two components of the fitness function
    term1 = sum((x - mu1)**2 for x in individual)
    term2 = d * n + s * sum((x - mu2)**2 for x in individual)
    term3 = 10 * sum(1 - np.cos(2 * np.pi * (x - mu1)) for x in individual)

    # Final fitness calculation
    fitness = min(term1, term2) + term3

    return fitness

n_dimensions = 2
# x0 = np.random.uniform(-5.12, 5.12, n_dimensions)
all_runs_local_optima = []
all_runs_edges = []
all_runs_fitness_values = []

class BasinHoppingCallback:
    def __init__(self, precision=1e-5, max_no_improve=1000):
        self.local_optima = []  # To track local optima
        self.fitness_values = []  # To track fitness values for the current run
        self.edges = {}  # Use a dictionary to store edges with their weights
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

        # Add or update edge
        if self.previous_optimum is not None:
            edge = (self.previous_optimum, rounded_x)
            if edge in self.edges:
                self.edges[edge] += 1  # Increment weight if edge exists
            else:
                self.edges[edge] = 1  # Initialize weight for new edge

        # Update previous optimum
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

    def get_lon_data(self):
        """
        Retrieve the collected data for constructing LONs or CLONs.

        Returns:
            tuple: (local_optima, fitness_values, edges)
                local_optima: List of unique local optima.
                fitness_values: List of fitness values corresponding to the local optima.
                edges: List of edges with weights as (source, target, weight).
        """
        edges_list = [(source, target, weight) for (source, target), weight in self.edges.items()]
        return self.local_optima, self.fitness_values, edges_list


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
for _ in range(1):  # Multiple runs
    x0 = np.random.uniform(-5.12, 5.12, n_dimensions)  # Random start
    callback = BasinHoppingCallback(max_no_improve=1000)
    result = basinhopping(
        birastrigin_eval,
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
print(all_runs_edges[0][0])

local_optima_data = [all_runs_local_optima, all_runs_fitness_values, all_runs_edges]

folder = 'data/birastriginN2A5'
filename = 'MonotonicSequenceBasinHopping'
save_filename = f'{filename}_LO.pkl'
save_path = os.path.join(folder, save_filename)

os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(local_optima_data, file)
    
print(f"Local optima saved to {save_path}")

def compress_lon(all_runs_local_optima, all_runs_fitness_values, all_runs_edges, accuracy=1e-4):
    """
    Compress a Monotonic LON (MLON) into a Compressed Monotonic LON (CMLON).

    Args:
        all_runs_local_optima (list): List of local optima nodes.
        all_runs_fitness_values (list): List of fitness values corresponding to the local optima.
        all_runs_edges (list of tuples): List of edges where each tuple is (source, target).
        accuracy (float): Accuracy threshold to group fitness values as equal.

    Returns:
        tuple: (compressed_nodes, compressed_edges, compressed_fitness_values)
            compressed_nodes: List of compressed local optima (sets of original nodes).
            compressed_edges: List of edges (source, target, count of transitions) for the CMLON.
            compressed_fitness_values: List of fitness values corresponding to the compressed nodes.
    """
    # Step 1: Group local optima by fitness value within the given accuracy
    compressed_nodes = []
    compressed_fitness_values = []
    fitness_to_node_map = {}

    for idx, fitness in enumerate(all_runs_fitness_values):
        found_group = False
        for key_fitness, group in fitness_to_node_map.items():
            if abs(fitness - key_fitness) <= accuracy:
                group.append(idx)
                found_group = True
                break
        if not found_group:
            fitness_to_node_map[fitness] = [idx]

    # Create compressed nodes as sets of original nodes
    for key_fitness, group in fitness_to_node_map.items():
        compressed_nodes.append(set(group))
        compressed_fitness_values.append(key_fitness)

    # Step 2: Aggregate edges between compressed nodes
    compressed_edges = {}

    for source, target in all_runs_edges:
        # Find the indices of source and target in all_runs_local_optima
        source_idx = next((i for i, node in enumerate(all_runs_local_optima) if np.allclose(node, source, atol=accuracy)), None)
        target_idx = next((i for i, node in enumerate(all_runs_local_optima) if np.allclose(node, target, atol=accuracy)), None)

        if source_idx is None or target_idx is None:
            continue

        # Find the compressed nodes containing the source and target
        source_group = next(group for group in compressed_nodes if source_idx in group)
        target_group = next(group for group in compressed_nodes if target_idx in group)

        # Create a unique key for the edge
        edge_key = (frozenset(source_group), frozenset(target_group))

        # Count the transitions
        if edge_key in compressed_edges:
            compressed_edges[edge_key] += 1
        else:
            compressed_edges[edge_key] = 1

    # Convert edge dictionary back to a list
    compressed_edges_list = [(list(edge[0]), list(edge[1]), count) for edge, count in compressed_edges.items()]

    return compressed_nodes, compressed_fitness_values, compressed_edges_list

compressed_local_optima_data = compress_lon(all_runs_local_optima[0], all_runs_fitness_values[0], all_runs_edges[0], accuracy=1e-5)
# compressed_local_optima_data = compress_lon(all_runs_local_optima, all_runs_fitness_values, all_runs_edges, accuracy=1e-5)


save_filename = f'{filename}_CLO.pkl'
save_path = os.path.join(folder, save_filename)

os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(local_optima_data, file)
print(f"Compressed Local optima saved to {save_path}")
