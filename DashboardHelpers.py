import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot2d_line(dataframe, value='final'):
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

def plot2d_box(dataframe, value='final'):
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

    noise_levels = sorted(df['noise'].unique())

    fig = px.box(
        df,
        x="noise",
        y="final_fit",
        color="algo_name",
        category_orders={"noise": noise_levels},
        points="all"  # optionally, to display all data points
    )

    fig.update_layout(
        boxmode="group",  # groups boxes for the same noise level
        xaxis_title="Noise",
        yaxis_title="Best solution found",
        legend_title="Algorithm",
        template="plotly_white"
    )
    # fig = go.Figure(data=[])
    return fig

def convert_to_rgba(color, opacity=1.0):
    from matplotlib.colors import to_rgba
    rgba = to_rgba(color, alpha=opacity)
    return f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})"

def fitness_to_color(fitness, min_fitness, max_fitness, alpha):
        # Normalize fitness value to a 0-1 scale
        if max_fitness == min_fitness:
            ratio = 0.5
        else:
            ratio = (fitness - min_fitness) / (max_fitness - min_fitness)
        # Define colors for the extremes
        low_rgb = np.array([0, 0, 255])   # Blue
        high_rgb = np.array([0, 255, 0])    # Green
        rgb = low_rgb + ratio * (high_rgb - low_rgb)
        rgb = rgb.astype(int)
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'

# Function to calculate Hamming distance
def hamming_distance(sol1, sol2):
    return sum(el1 != el2 for el1, el2 in zip(sol1, sol2))

def select_top_runs_by_fitness(all_run_trajectories, n_runs_display, optimisation_goal):
    if optimisation_goal == 'max':
        # Sort runs by the best (highest) final fitness, in descending order
        sorted_runs = sorted(all_run_trajectories, 
                            key=lambda run: run[1][-1],
                            reverse=True)
    else:
        # Sort runs by the best (lowest) final fitness, in ascending order
        sorted_runs = sorted(all_run_trajectories, 
                            key=lambda run: run[1][-1],
                            reverse=False)
    top_runs = sorted_runs[:n_runs_display] # Cap the number of runs to display
    return top_runs

def get_mean_run(all_run_trajectories):
    final_fitnesses = [run[1][-1] for run in all_run_trajectories]
    mean_final_fitness = np.mean(final_fitnesses)
    closest_run_idx = np.argmin([abs(fitness - mean_final_fitness) for fitness in final_fitnesses])
    return all_run_trajectories[closest_run_idx]

def get_median_run(all_run_trajectories):
    final_fitnesses = [run[1][-1] for run in all_run_trajectories]
    median_final_fitness = np.median(final_fitnesses)
    closest_run_idx = np.argmin([abs(fitness - median_final_fitness) for fitness in final_fitnesses])
    return all_run_trajectories[closest_run_idx]

def determine_optimisation_goal(all_trajectories_list):
        first_run = all_trajectories_list[0][0]  # Access the first trajectory
        starting_fitness = first_run[1][0]  # Initial fitness value
        ending_fitness = first_run[1][-1]  # Final fitness value
        # print(starting_fitness)
        # print(ending_fitness)
        return "min" if ending_fitness < starting_fitness else "max"

# ===========
# Data
# ==========

def convert_to_split_edges_format(data):
    """
    Convert a compressed LON dictionary with `edges` to a format with separate `edge_transitions` and `edge_weights`.

    Args:
        data (dict): Original compressed LON data with keys:
            - "local_optima": List of unique local optima.
            - "fitness_values": List of fitness values corresponding to the local optima.
            - "edges": Dictionary of edges with weights {(source, target): weight}.
    Returns:
        dict: Modified compressed LON data with `edge_transitions` and `edge_weights` instead of `edges`.
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edge_transitions": [],
        "edge_weights": [],
    }

    for (source, target), weight in data["edges"].items():
        converted_data["edge_transitions"].append((source, target))
        converted_data["edge_weights"].append(weight)

    return converted_data

def convert_to_single_edges_format(data):
    """
    Convert a compressed LON dictionary with separate `edge_transitions` and `edge_weights` to a format with `edges`.

    Args:
        data (dict): Modified compressed LON data with keys:
            - "local_optima": List of unique local optima.
            - "fitness_values": List of fitness values corresponding to the local optima.
            - "edge_transitions": List of edges as (source, target).
            - "edge_weights": List of edge weights corresponding to transitions.

    Returns:
        dict: Original compressed LON data with `edges` instead of `edge_transitions` and `edge_weights`.
    """
    converted_data = {
        "local_optima": data["local_optima"],
        "fitness_values": data["fitness_values"],
        "edges": {},
    }

    # print("Edge Transitions Sample:", data["edge_transitions"][:])
    # print("Edge Transition Types:", [type(transition) for transition in data["edge_transitions"][:]])

    for transition, weight in zip(data["edge_transitions"], data["edge_weights"]):
        # Ensure transition is a valid list or tuple with two elements
        if isinstance(transition, (list, tuple)) and len(transition) == 2:
            source, target = map(tuple, transition)  # Convert source and target to tuples
            converted_data["edges"][(source, target)] = weight
        else:
            raise ValueError(f"Invalid transition format: {transition}")


    # # Ensure edge_transitions are tuples
    # for transition, weight in zip(data["edge_transitions"], data["edge_weights"]):
    #     source, target = map(tuple, transition) if isinstance(transition, list) else transition
    #     converted_data["edges"][(source, target)] = weight

    return converted_data

def quadratic_bezier(start, end, curvature=0.2, n_points=20):
    """
    Computes a set of points for a quadratic Bézier curve between start and end.
    curvature: fraction of the distance to offset the midpoint perpendicular to the line.
    n_points: number of points along the curve.
    """
    start = np.array(start)
    end = np.array(end)
    # Compute the midpoint.
    mid = (start + end) / 2.0
    
    # Compute the perpendicular vector.
    direction = end - start
    if np.all(direction == 0):
        # Prevent division by zero if start equals end.
        return np.array([start])
    perp = np.array([-direction[1], direction[0]])
    perp = perp / np.linalg.norm(perp)
    
    # Offset the midpoint by a fraction of the distance.
    distance = np.linalg.norm(direction)
    control = mid + curvature * distance * perp
    
    # Generate points along the quadratic Bézier curve.
    t_values = np.linspace(0, 1, n_points)
    curve_points = []
    for t in t_values:
        point = (1 - t)**2 * start + 2 * (1 - t) * t * control + t**2 * end
        curve_points.append(point)
    
    return np.array(curve_points)