
def get_local_optima(folder, filename, maximisation=True, filtering='same', similarity=1e-2):
    import os
    import pickle
    import numpy as np

    load_path = os.path.join(folder, f'{filename}.pkl')

    with open(load_path, 'rb') as file:
        all_run_trajectories = pickle.load(file)

    final_unique_solutions = [run[0][-1] for run in all_run_trajectories] # unique_solutions
    final_unique_fitnesses = [run[1][-1] for run in all_run_trajectories] # unique_fitnesses

    solution_fitness_pairs = list(zip(final_unique_solutions, final_unique_fitnesses))
    solution_fitness_pairs.sort(key=lambda x: x[1], reverse=maximisation)

    if filtering == 'similar':
        # Filter out solutions that are similar in both elements and fitness
        filtered_pairs = []
        for solution, fitness in solution_fitness_pairs:
            if not any(
                abs(fitness - kept_fitness) < similarity and
                np.allclose(solution, kept_solution, atol=similarity)
                for kept_solution, kept_fitness in filtered_pairs
            ):
                filtered_pairs.append((solution, fitness))
        final_unique_solutions_no_duplicates = [pair[0] for pair in filtered_pairs]
        final_unique_fitnesses_no_duplicates = [pair[1] for pair in filtered_pairs]

    elif filtering == 'same':
        # Filter out solutions that are identical
        unique_pairs = {str(solution): (solution, fitness) for solution, fitness in solution_fitness_pairs}
        final_unique_solutions_no_duplicates = [pair[0] for pair in unique_pairs.values()]
        final_unique_fitnesses_no_duplicates = [pair[1] for pair in unique_pairs.values()]

    # Print max and min fitness info
    max_fitness = max(final_unique_fitnesses_no_duplicates)
    min_fitness = min(final_unique_fitnesses_no_duplicates)
    print(f"Maximum fitness: {max_fitness}")
    print(f"Minimum fitness: {min_fitness}")

    Local_optima = [final_unique_solutions_no_duplicates, final_unique_fitnesses_no_duplicates]
    save_filename = f'{filename}_filter{filtering}_LO.pkl'
    save_path = os.path.join(folder, save_filename)

    with open(save_path, 'wb') as file:
        pickle.dump(Local_optima, file)
    
    print(f"Local optima saved to {save_path}")

# get_local_optima('data/rastriginN2A10', 'HC_g10000_pNA_20241209021744')
get_local_optima('data/knapPI_3_100_1000_1', 'HC_g100000_pNA_20241210212912', maximisation=True, filtering='same')