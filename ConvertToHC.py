
def get_local_optima(folder, filename):
    import os
    import pickle

    load_path = os.path.join(folder, f'{filename}.pkl')

    with open(load_path, 'rb') as file:
        all_run_trajectories = pickle.load(file)

    final_unique_solutions = [run[0][-1] for run in all_run_trajectories] # unique_solutions
    final_unique_fitnesses = [run[1][-1] for run in all_run_trajectories] # unique_fitnesses

    solution_fitness_pairs = list(zip(final_unique_solutions, final_unique_fitnesses))
    unique_pairs = {str(solution): (solution, fitness) for solution, fitness in solution_fitness_pairs}
    final_unique_solutions_no_duplicates = [pair[0] for pair in unique_pairs.values()]
    final_unique_fitnesses_no_duplicates = [pair[1] for pair in unique_pairs.values()]

    Local_optima = [final_unique_solutions_no_duplicates, final_unique_fitnesses_no_duplicates]

    save_filename = f'{filename}_LO.pkl'
    save_path = os.path.join(folder, save_filename)

    with open(save_path, 'wb') as file:
        pickle.dump(Local_optima, file)
    
    print(f"Local optima saved to {save_path}")

get_local_optima('data/rastriginN2A10', 'HC_g10000_pNA_20241209021744')