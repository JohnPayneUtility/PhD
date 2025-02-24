import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('results.pkl')
# print(df.head(10))
print(df.columns)

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

    # increase linewidth for PCEA to avoid overlap with UMDA
    # if 'PCEA' in algo:
    #     lw = 3  # increased line width for better visibility
    # else:
    lw = 1.5  # default line width for others
    
    plt.errorbar(
        subset['noise'], 
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