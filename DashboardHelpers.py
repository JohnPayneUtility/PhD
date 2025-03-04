import plotly.graph_objects as go
import pandas as pd

def plot2d(dataframe, value='final'):
    # assert "my_column" in df.columns, "DataFrame must contain column 'my_column'"
    # group_columns = ['algo_name', 'noise']
    df = dataframe.copy()
    df = df[['algo_name', 'noise', 'final_fit']]

    if value == 'final':
        stats = df.groupby(['algo_name', 'noise'])['final_fit'].agg(['mean', 'std']).reset_index()
    
    # Determine run count
    # first_algo = stats['algo_name'].iloc[0]
    # first_algo_df = stats[stats['algo_name'] == first_algo]
    # run_counts = first_algo_df.size()
    # runs_per_group = run_counts.iloc[0]

    fig = go.Figure()
    for algo in stats['algo_name'].unique():
        subset = stats[stats['algo_name'] == algo]
        fig.add_trace(go.Scatter(
            x=subset['noise'],
            y=subset['mean'],
            error_y=dict(
                type='data',
                array=subset['std'],
                visible=True,
                thickness=1.5,
                width=5
            ),
            mode='lines+markers',
            name=algo
        ))

    fig.update_layout(
        # title=f'Comparison of algorithms for {df["problem_name"].unique()[0]} (opt: {df["opt_global"].iloc[0]}) for different noise levels ({runs_per_group} runs)',
        title = 'title',
        xaxis_title=r'$\sigma$ (Standard Deviation of Gaussian Noise $N(0,\sigma)$)',
        yaxis_title='Best solution found',
        legend_title='Algo Name',
        template='plotly_white'
    )
    # fig = go.Figure(data=[])
    return fig