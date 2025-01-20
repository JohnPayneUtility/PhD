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
            dict: Dictionary containing LON data with the following keys:
                - 'local_optima': List of unique local optima.
                - 'fitness_values': List of fitness values corresponding to the local optima.
                - 'edges': Dictionary of edges with their weights {(source, target): weight}.
        """
        edges_list = [(source, target, weight) for (source, target), weight in self.edges.items()]
        return self.local_optima, self.fitness_values, edges_list

# Problem information and data initialisation
n_dimensions = 2
# x0 = np.random.uniform(-5.12, 5.12, n_dimensions)
all_runs_local_optima = []
all_runs_edges = []
all_runs_fitness_values = []

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
aggregated_lon_data = {
    "local_optima": [],
    "fitness_values": [],
    "edges": {},
}

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
    local_optima, fitness_values, edges_list = callback.get_lon_data()

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

# Extract local optima and filter duplicates
# local_optima = set(tuple(np.round(res.x, 5)) for res in results)
# local_optima = set(tuple(res.x) for res in results)

# Print results
# print(f"Number of unique local optima: {len(local_optima)}")
# for opt in local_optima:
#     print(f"Optimum: {opt}")

# print(all_runs_local_optima)
# print(all_runs_fitness_values)
# print(len(all_runs_local_optima))
# print(len(all_runs_local_optima[0]))
# print(all_runs_edges[0][0])

# local_optima_data = [all_runs_local_optima, all_runs_fitness_values, all_runs_edges]

# Save local optima data for runs
folder = 'data/birastriginN2A5'
filename = 'MonotonicSequenceBasinHopping'
save_filename = f'{filename}_LO.pkl'
save_path = os.path.join(folder, save_filename)
os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(aggregated_lon_data, file)
print(f"Local optima saved to {save_path}")

print(aggregated_lon_data["local_optima"][0])
print(aggregated_lon_data["fitness_values"][0])
print(next(iter(aggregated_lon_data["edges"].items())))

# def compress_lon(all_runs_local_optima, all_runs_fitness_values, all_runs_edges, accuracy=1e-4):
#     """
#     Compress a Monotonic LON (MLON) into a Compressed Monotonic LON (CMLON).

#     Args:
#         all_runs_local_optima (list): List of local optima nodes.
#         all_runs_fitness_values (list): List of fitness values corresponding to the local optima.
#         all_runs_edges (list of tuples): List of edges where each tuple is (source, target).
#         accuracy (float): Accuracy threshold to group fitness values as equal.

#     Returns:
#         tuple: (compressed_nodes, compressed_edges, compressed_fitness_values)
#             compressed_nodes: List of compressed local optima (sets of original nodes).
#             compressed_edges: List of edges (source, target, count of transitions) for the CMLON.
#             compressed_fitness_values: List of fitness values corresponding to the compressed nodes.
#     """
#     # Step 1: Group local optima by fitness value within the given accuracy
#     compressed_nodes = []
#     compressed_fitness_values = []
#     fitness_to_node_map = {}

#     for idx, fitness in enumerate(all_runs_fitness_values):
#         found_group = False
#         for key_fitness, group in fitness_to_node_map.items():
#             if abs(fitness - key_fitness) <= accuracy:
#                 group.append(idx)
#                 found_group = True
#                 break
#         if not found_group:
#             fitness_to_node_map[fitness] = [idx]

#     # Create compressed nodes as sets of original nodes
#     for key_fitness, group in fitness_to_node_map.items():
#         compressed_nodes.append(set(group))
#         compressed_fitness_values.append(key_fitness)

#     # Step 2: Aggregate edges between compressed nodes
#     compressed_edges = {}

#     for source, target in all_runs_edges:
#         # Find the indices of source and target in all_runs_local_optima
#         source_idx = next((i for i, node in enumerate(all_runs_local_optima) if np.allclose(node, source, atol=accuracy)), None)
#         target_idx = next((i for i, node in enumerate(all_runs_local_optima) if np.allclose(node, target, atol=accuracy)), None)

#         if source_idx is None or target_idx is None:
#             continue

#         # Find the compressed nodes containing the source and target
#         source_group = next(group for group in compressed_nodes if source_idx in group)
#         target_group = next(group for group in compressed_nodes if target_idx in group)

#         # Create a unique key for the edge
#         edge_key = (frozenset(source_group), frozenset(target_group))

#         # Count the transitions
#         if edge_key in compressed_edges:
#             compressed_edges[edge_key] += 1
#         else:
#             compressed_edges[edge_key] = 1

#     # Convert edge dictionary back to a list
#     compressed_edges_list = [(list(edge[0]), list(edge[1]), count) for edge, count in compressed_edges.items()]

#     return compressed_nodes, compressed_fitness_values, compressed_edges_list

# compressed_local_optima_data = compress_lon(all_runs_local_optima[0], all_runs_fitness_values[0], all_runs_edges[0], accuracy=1e-5)
# compressed_local_optima_data = compress_lon(all_runs_local_optima, all_runs_fitness_values, all_runs_edges, accuracy=1e-5)

def compress_lon(aggregated_lon_data, accuracy=1e-5):
    """
    Compress a Landscape of Optima Network (LON) by condensing nodes with similar fitness values.

    Args:
        aggregated_lon_data (dict): Dictionary containing LON data with keys:
            - "local_optima": List of unique local optima.
            - "fitness_values": List of fitness values corresponding to the local optima.
            - "edges": Dictionary of edges with their weights {(source, target): weight}.
        accuracy (float): Accuracy threshold for grouping nodes with similar fitness values.

    Returns:
        dict: Compressed LON data in the same format as the input.
    """
    # Initialize compressed LON data
    compressed_lon_data = {
        "local_optima": [],
        "fitness_values": [],
        "edges": {},
    }

    # Map to track which local optima are grouped together
    fitness_to_group = {}
    group_to_optima = []

    # Group nodes by fitness values within the given accuracy
    for opt, fitness in zip(aggregated_lon_data["local_optima"], aggregated_lon_data["fitness_values"]):
        found_group = False
        for group_idx, group_fitness in enumerate(compressed_lon_data["fitness_values"]):
            if abs(fitness - group_fitness) <= accuracy:
                # Add to existing group
                fitness_to_group[opt] = group_idx
                group_to_optima[group_idx].append(opt)
                found_group = True
                break
        if not found_group:
            # Create a new group
            group_idx = len(compressed_lon_data["local_optima"])
            compressed_lon_data["local_optima"].append(opt)
            compressed_lon_data["fitness_values"].append(fitness)
            group_to_optima.append([opt])
            fitness_to_group[opt] = group_idx

    # Aggregate edges between compressed nodes
    for (source, target), weight in aggregated_lon_data["edges"].items():
        # print(source)
        # print(target)
        # print(weight)
        source_group = fitness_to_group[source]
        target_group = fitness_to_group[target]

        # Create an edge between the compressed groups
        compressed_edge = (source_group, target_group)
        if compressed_edge in compressed_lon_data["edges"]:
            compressed_lon_data["edges"][compressed_edge] += weight
        else:
            compressed_lon_data["edges"][compressed_edge] = weight

    return compressed_lon_data

compressed_lon_data = compress_lon(aggregated_lon_data, accuracy=1e-4)
save_filename = f'{filename}_CLO.pkl'
save_path = os.path.join(folder, save_filename)
os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(compressed_lon_data, file)
print(f"Compressed Local optima saved to {save_path}")

print(len(aggregated_lon_data["local_optima"]))
print(len(compressed_lon_data["local_optima"]))