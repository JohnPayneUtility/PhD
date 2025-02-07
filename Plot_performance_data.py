import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.head())

# Create aggregated dataframe
group_columns = ['problem_name', 'algo_name', 'Noise_type', 'Noise_intensity']
aggregated_df = df.groupby(group_columns, as_index=False)[['Max_fitness', 'Min_fitness', 'Final_fitness']].mean()

# count runs
first_algo = df['algo_name'].iloc[0]
first_algo_df = df[df['algo_name'] == first_algo]
grouped_first_algo_df = first_algo_df.groupby(group_columns)
run_counts = grouped_first_algo_df.size()
runs_per_group = run_counts.iloc[0]
print(runs_per_group)

print(aggregated_df.head())

algos = df['algo_name'].unique()
stats = df.groupby(['algo_name', 'Noise_intensity'])['Max_fitness'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
# Loop through each unique algorithm and plot its data with error bars
for algo in stats['algo_name'].unique():
    subset = stats[stats['algo_name'] == algo]

    # increase linewidth for PCEA to avoid overlap with UMDA
    if 'PCEA' in algo:
        lw = 3  # increased line width for better visibility
    else:
        lw = 1.5  # default line width for others
    
    plt.errorbar(
        subset['Noise_intensity'], 
        subset['mean'], 
        yerr=subset['std'], 
        fmt='-o',   # line with markers
        capsize=5,  # adds caps to the error bars for better visibility
        label=algo,
        linewidth=lw
    )

# Customize the plot
plt.xlabel(r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)')
plt.ylabel('Best solution found')
plt.title(f'Comparison of algorithms for OneMax for different noise levels ({runs_per_group} runs)')
plt.legend(title='Algo Name')
plt.grid(True)
plt.show()

stats = df.groupby(['algo_name', 'Noise_intensity'])['Final_fitness'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(10, 6))
# Loop through each unique algorithm and plot its data with error bars
for algo in stats['algo_name'].unique():
    subset = stats[stats['algo_name'] == algo]

    # increase linewidth for PCEA to avoid overlap with UMDA
    if 'PCEA' in algo:
        lw = 3  # increased line width for better visibility
    else:
        lw = 1.5  # default line width for others
    
    plt.errorbar(
        subset['Noise_intensity'], 
        subset['mean'], 
        yerr=subset['std'], 
        fmt='-o',   # line with markers
        capsize=5,  # adds caps to the error bars for better visibility
        label=algo,
        linewidth=lw
    )

# Customize the plot
plt.xlabel(r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)')
plt.ylabel('Final solution found')
plt.title(f'Comparison of algorithms for OneMax for different noise levels ({runs_per_group} runs)')
plt.legend(title='Algo Name')
plt.grid(True)
plt.show()

