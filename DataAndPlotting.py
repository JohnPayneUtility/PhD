import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------
# Knapsack Problem Data
# ----------------------------------------------------------------

def get_kp_files(base_folder: str = "instances_01_KP") -> list:
    """
    Creates list of tuples containing (filename, global_optimum) from KP problem files
    """
    subfolders = ["low-dimensional-optimum", "large_scale-optimum"]
    results = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(folder_path):
            # print('no path')
            continue  # Skip if the subfolder does not exist
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Read the file content (a single number)
            with open(file_path, "r") as f:
                content = f.read()
            # Convert to int or float
            try:
                optimum = int(content)
            except ValueError:
                optimum = float(content)
            results.append((filename, optimum))
    return results



# ----------------------------------------------------------------
# Problem Performance Plotting
# ----------------------------------------------------------------


input_df = pd.read_pickle('results.pkl')
# print(df.head(10))
print(input_df.columns)

unique_problems = input_df['problem_name'].unique()
for problem in unique_problems:
    df = input_df[input_df['problem_name'] == problem]
    opt_global = df["opt_global"].iloc[0]

    group_columns = ['problem_name', 'algo_name', 'noise']
    aggregated_df = df.groupby(group_columns, as_index=False)[['max_fit', 'min_fit', 'final_fit']].mean()

    first_algo = df['algo_name'].iloc[0]
    first_algo_df = df[df['algo_name'] == first_algo]
    grouped_first_algo_df = first_algo_df.groupby(group_columns)
    run_counts = grouped_first_algo_df.size()
    runs_per_group = run_counts.iloc[0]
    print(runs_per_group)

    print(aggregated_df.head())

    algos = df['algo_name'].unique()
    stats = df.groupby(['algo_name', 'noise'])['final_fit'].agg(['mean', 'std']).reset_index()

    print(stats.head(100))

    plt.figure(figsize=(10, 6))
    # Loop through each unique algorithm and plot its data with error bars
    for algo in stats['algo_name'].unique():
        subset = stats[stats['algo_name'] == algo]

        plt.errorbar(
            subset['noise'], 
            subset['mean'], 
            yerr=subset['std'], 
            fmt='-o',   # line with markers
            capsize=5,  # adds caps to the error bars for better visibility
            label=algo,
            linewidth=1.5
        )

    # Customize the plot
    plt.xlabel(r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)')
    plt.ylabel('Best solution found')
    plt.title(f'Comparison of algorithms for {problem} (opt:{opt_global}) for different noise levels ({runs_per_group} runs)')
    plt.legend(title='Algo Name')
    plt.grid(True)
    plt.show()