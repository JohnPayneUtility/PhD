import numpy as np
from scipy.optimize import basinhopping, minimize
import os
import pickle
from LON_Utilities import convert_to_split_edges_format
from FitnessFunctions import *

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
n_dimensions = 5
base_step_size = 0.4749
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

for i in range(100):  # Multiple runs
    x0 = np.random.uniform(-5.12, 5.12, n_dimensions)  # Random start
    callback = BasinHoppingCallback(max_no_improve=1000)
    result = basinhopping(
        rastrigin_eval,
        x0=np.random.uniform(-5.12, 5.12, n_dimensions),  # Random start
        minimizer_kwargs=minimizer_kwargs,
        niter=10000,  # Maximum basin-hopping iterations
        stepsize=(2*base_step_size),
        callback=callback,
    )
    results.append(result)
    print(i)

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
aggregated_lon_data_SE = convert_to_split_edges_format(aggregated_lon_data)
folder = 'data/RastriginN5A10'
filename = 'MonotonicSequenceBasinHopping'
save_filename = f'{filename}_LO.pkl'
save_path = os.path.join(folder, save_filename)
os.makedirs(folder, exist_ok=True)
with open(save_path, 'wb') as file:
        pickle.dump(aggregated_lon_data_SE, file)
print(f"Local optima saved to {save_path}")

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

print(len(aggregated_lon_data["local_optima"]))
print(len(compressed_lon_data["local_optima"]))

# print(aggregated_lon_data)

print("Local Optima Sample:", aggregated_lon_data["local_optima"][:3])
print("Fitness Values Sample:", aggregated_lon_data["fitness_values"][:3])
# Check edges or edge transitions and weights
if "edges" in aggregated_lon_data:
    print("Edges Sample:", list(aggregated_lon_data["edges"].items())[:3])
elif "edge_transitions" in aggregated_lon_data and "edge_weights" in aggregated_lon_data:
    print("Edge Transitions Sample:", aggregated_lon_data["edge_transitions"][:3])
    print("Edge Weights Sample:", aggregated_lon_data["edge_weights"][:3])