import os
import pickle

def delete_hc_files(directory):
    """Delete all .pkl files starting with 'HC' in a directory and its subdirectories."""
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("HC") and filename.endswith(".pkl"):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def transform_hc_data(directory):
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("HC") and filename.endswith(".pkl") and not filename.endswith("LO.pkl"):
                file_path = os.path.join(dirpath, filename)
                try:
                    # Load the .pkl file
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)

                    # Check if the data format is a list of runs
                    if not isinstance(data, list):
                        print(f"Unexpected format in {filename}. Content is not a list.")
                        continue

                    # Initialize aggregated lists
                    unique_solutions_all = []
                    unique_fitnesses_all = []
                    solution_iterations_all = []
                    transitions_all = []

                    # Process each run
                    for run in data:
                        if len(run) == 4:  # Ensure the run has the expected structure
                            unique_solutions, unique_fitnesses, solution_iterations, transitions = run
                            unique_solutions_all.append(unique_solutions)
                            unique_fitnesses_all.append(unique_fitnesses)
                            solution_iterations_all.append(solution_iterations)
                            transitions_all.append(transitions)
                        else:
                            print(f"Run format issue in {filename}. Skipping run: {run}")

                    # Combine all aggregated lists into the new structure
                    transformed_data = [
                        unique_solutions_all,
                        unique_fitnesses_all,
                        transitions_all
                    ]

                    # Save the new data
                    new_file_name = filename.replace(".pkl", "_LO.pkl")
                    new_file_path = os.path.join(dirpath, new_file_name)
                    with open(new_file_path, 'wb') as f:
                        pickle.dump(transformed_data, f)

                    print(f"Transformed and saved: {new_file_path}")

                except EOFError:
                    print(f"File {filename} in {dirpath} is empty or corrupted.")
                except pickle.UnpicklingError:
                    print(f"Error unpickling {filename} in {dirpath}. File might be invalid.")
                except Exception as e:
                    print(f"Error processing file {filename} in {dirpath}: {e}")

# Call the function with the target directory
transform_hc_data('./data')
# delete_hc_files('data')
