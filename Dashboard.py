import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import TSNE

import os
import pickle
import json

# Get list of folders in the data directory
data_folder = 'data'
folder_options = [{'label': folder, 'value': folder} for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

def determine_optimisation_goal(all_trajectories_list):
        first_run = all_trajectories_list[0][0]  # Access the first trajectory
        starting_fitness = first_run[1][0]  # Initial fitness value
        ending_fitness = first_run[1][-1]  # Final fitness value
        # print(starting_fitness)
        # print(ending_fitness)
        return "min" if ending_fitness < starting_fitness else "max"

algo_colors = ['blue', 'orange', 'purple', 'brown', 'cyan', 'magenta']

# Create Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Search Trajectory Network Dashboard"),
    html.P(
        "This plot visualises the search trajectories of algorithms for optimisation problems. "
        "Each trajectory is a squence of solutions corresponding to the best solution in the population over time. "
        "Each node corresponds to the location of a best solution in the trajectory. "
        "Edges are directed and connect two consecutive locations of best solutions in the search trajectory. ",
        style={'fontSize': 16, 'marginTop': '10px'}
    ),
    html.Div([
        html.Label("Select problem:", style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='folder-dropdown',
            options=folder_options,
            value=folder_options[0]['value'] if folder_options else None,
            style={'width': '200px'}
        ),
    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '20px'}),
    html.P(
        id='display-problem-info',
        style={'fontSize': 16, 'marginTop': '10px'}
    ),
    html.Label("Select optima data:"),
    dcc.Dropdown(
        id='optima-file-dropdown',
        multi=False
    ),
    dcc.Store(id='local-optima'),
    html.Label("Select algorithm data:"),
    dcc.Dropdown(
        id='file-dropdown',
        multi=True
    ),
    dcc.Store(id='loaded-data-store'),
    html.Div([
        html.Label("Select number of runs to show:"),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Div([
        dcc.Input(
            id='run-selector',
            type='number',
            min=0,
            max=1000,
            step=1,
            value=1
        ),
    ], style={'display': 'inline-block'}),
    dcc.Checklist(
        id='run-options',
        options=[
            {'label': 'Show best run', 'value': 'show_best'},
            {'label': 'Show mean runs', 'value': 'show_mean'},
            {'label': 'Show median runs', 'value': 'show_median'},
            {'label': 'Show worst run', 'value': 'show_worst'}
        ],
        value=[]
    ),
    html.Hr(),
    dcc.Checklist(
        id='options',
        options=[
            {'label': 'Show Labels', 'value': 'show_labels'},
            {'label': 'Hide Nodes', 'value': 'hide_nodes'},
            {'label': '3D Plot', 'value': 'plot_3D'},
            {'label': 'Use Solution Iterations', 'value': 'use_solution_iterations'}
        ],
        value=[]
    ),
    dcc.Dropdown(
        id='layout',
        options=[
            {'label': 'Fruchterman Reignold force directed', 'value': 'spring'},
            {'label': 'Kamada Kawai force directed', 'value': 'kamada_kawai'},
            {'label': 'MDS dissimilarity', 'value': 'mds'},
            {'label': 't-SNE dissimilarity', 'value': 'tsne'},
            {'label': 'raw solution values', 'value': 'raw'}
        ],
        value='kamada_kawai',
        placeholder='Select a layout',
        style={'width': '50%', 'marginTop': '10px'}
    ),
    dcc.Dropdown(
        id='hover-info',
        options=[
            {'label': 'Show Fitness', 'value': 'fitness'},
            {'label': 'Show Iterations', 'value': 'iterations'},
            {'label': 'Show solutions', 'value': 'solutions'}
        ],
        value='fitness',
        placeholder='Select hover information',
        style={'width': '50%', 'marginTop': '10px'}
    ),
    dcc.Checklist(
        id='use-range-sliders',
        options=[
            {'label': 'limit axis ranges', 'value': 'ENABLED'}
        ],
        value=[]  # Default is unchecked
    ),
    html.Div(
    [
        html.Label("X-axis range"),
        dcc.Slider(
            id='x-axis-slider',
            min=1,
            max=10,
            step=0.1,
            value=5,  # Default range
            marks={i: f"{i}" for i in range(1, 11)}
        )
    ],
    id='x-slider-container',  # Add an ID for the container
    style={'display': 'none'}  # Initially hidden
    ),
    html.Div(
        [
            html.Label("Y-axis range"),
            dcc.Slider(
                id='y-axis-slider',
                min=1,
                max=10,
                step=0.1,
                value=5,  # Default range
                marks={i: f"{i}" for i in range(1, 11)}
            )
        ],
        id='y-slider-container',  # Add an ID for the container
        style={'display': 'none'}  # Initially hidden
    ),
    dcc.Graph(id='trajectory-plot'),
    dcc.Store(id='algo-info'),
    html.Div(id='display-algo-info', style={'marginTop': '10px'}),
    html.P(id='unique-solutions-display', children='', style={'marginTop': '20px'})
])

@app.callback(
    [Output('x-slider-container', 'style'),
     Output('y-slider-container', 'style')],
    Input('use-range-sliders', 'value')
)
def toggle_range_elements(value):
    if 'ENABLED' in value:
        return {'display': 'block'}, {'display': 'block'}  # Show both containers
    return {'display': 'none'}, {'display': 'none'}  # Hide both containers

# File selection
@app.callback(
    [Output('file-dropdown', 'options'),
    Output('display-problem-info', 'children')],
    Input('folder-dropdown', 'value')
)
def update_file_dropdown(selected_folder):
    if selected_folder:
        folder_path = os.path.join(data_folder, selected_folder)
        file_options = [{'label': file, 'value': file} for file in os.listdir(folder_path) if file.endswith('.pkl') and not file.endswith('LO.pkl')]
        
        # Try to read 'info.txt' if it exists in the folder
        info_file_path = os.path.join(folder_path, 'info.txt')
        if os.path.exists(info_file_path):
            with open(info_file_path, 'r') as info_file:
                info_content = info_file.read()
        else:
            info_content = "Problem info not available"

        return file_options, info_content
    return [], "No problem selected"

@app.callback(
    Output('optima-file-dropdown', 'options'),
    Input('folder-dropdown', 'value')
)
def update_optima_file_dropdown(selected_folder):
    if selected_folder:
        folder_path = os.path.join(data_folder, selected_folder)
        file_options = [{'label': file, 'value': file} for file in os.listdir(folder_path) if file.endswith('LO.pkl')]
        return file_options
    return []

@app.callback(
    Output('local-optima', 'data'),
    [Input('folder-dropdown', 'value'),
     Input('optima-file-dropdown', 'value')]
)
def load_optima_data(selected_folder, selected_file):
    if not selected_folder or not selected_file:
        return None
    
    file_path = os.path.join(data_folder, selected_folder, selected_file)
    with open(file_path, 'rb') as f:
        local_optima = pickle.load(f)

    return local_optima

# Data loading
@app.callback(
    [Output('loaded-data-store', 'data'),
     Output('algo-info', 'data'),
     Output('unique-solutions-display', 'children')],
    [Input('folder-dropdown', 'value'),
     Input('file-dropdown', 'value')]
)
def load_data(selected_folder, selected_files):
    if not selected_folder or not selected_files:
        return None, [], []

    all_trajectories_list = []
    all_info_files_list = []

    for file_name in selected_files:
        # Load trajectory data (pickle file)
        file_path = os.path.join(data_folder, selected_folder, file_name)
        with open(file_path, 'rb') as f:
            all_trajectories_list.append(pickle.load(f))
        
        # Load JSON parameter file
        info_filename = file_name.replace('pkl', 'json') # replacer pkl with json for parameter file of same name
        info_file_path = os.path.join(data_folder, selected_folder, info_filename)
        if os.path.exists(info_file_path):
            with open(info_file_path, 'r') as info_file:
                all_info_files_list.append(json.load(info_file))
        else:
            all_info_files_list.append({"error": "Algo info not available"})

    # Generate unique solutions text
    unique_solutions_text = str([
        (f"Algorithm {idx + 1} - First Run Unique Solutions:\n{trajectories[0][1]}\n\n", idx)
        for idx, trajectories in enumerate(all_trajectories_list) if trajectories
    ])

    return all_trajectories_list, all_info_files_list, unique_solutions_text

@app.callback(
    Output('display-algo-info', 'children'),
    Input('algo-info', 'data')
)
def update_algo_info(all_info_files_list):
    def format_algo_info(info, color):
        # Format the dictionary as a list of key-value pairs
        formatted_info = [
            html.Div(
                children=[
                    html.Span(f"{key}: ", style={'fontWeight': 'bold'}),
                    html.Span(str(value))
                ]
            )
            for key, value in info.items()
        ]
        return html.Div(
            children=formatted_info,
            style={'color': color, 'fontSize': '14px', 'marginBottom': '10px'}
        )

    # Loop through each dictionary and format it
    info_divs = [
        html.Div(
            children=[
                html.Span(f"Algorithm {i + 1} Info:", style={'fontWeight': 'bold', 'color': algo_colors[i]}),
                format_algo_info(info, algo_colors[i])
            ],
            style={'marginBottom': '10px'}
        )
        for i, info in enumerate(all_info_files_list)
    ]

    return info_divs

# Plotting
@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('options', 'value'),
     Input('run-options', 'value'),
     Input('layout', 'value'),
     Input('hover-info', 'value'),
     Input('loaded-data-store', 'data'),
     Input('run-selector', 'value'),
     Input('local-optima', 'data'),
     Input('use-range-sliders', 'value'),
     Input('x-axis-slider', 'value'),
    Input('y-axis-slider', 'value')]
)
def update_plot(options, run_options, layout_value, hover_info_value, all_trajectories_list, n_runs_display, local_optima, use_range_slider, x_slider, y_slider):
    # if not all_trajectories_list:
    #     return go.Figure()
    
    # Options from checkboxes
    show_labels = 'show_labels' in options
    hide_nodes = 'hide_nodes' in options
    plot_3D = 'plot_3D' in options
    use_solution_iterations = 'use_solution_iterations' in options

    # Run options
    show_best = 'show_best' in run_options
    show_mean = 'show_mean' in run_options
    show_median = 'show_median' in run_options
    show_worst = 'show_worst' in run_options

    # Options from dropdowns
    layout = layout_value

    G = nx.DiGraph()

    # Colors for different sets of trajectories
    # algo_colors = ['blue', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    node_color_shared = 'green'
    local_optima_color = 'black'

    # Add nodes and edges for each set of trajectories
    node_mapping = {}  # To ensure unique solutions map to the same node
    start_nodes = set()
    end_nodes = set()
    overall_best_fitness = 0
    overall_best_node = None

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
    
    transition_counts = {}
    # Function to add nodes and edges to the graph
    def add_trajectories_to_graph(all_run_trajectories, edge_color):
        for run_idx, (unique_solutions, unique_fitnesses, solution_iterations, transitions) in enumerate(all_run_trajectories):
            for i, solution in enumerate(unique_solutions):
                solution_tuple = tuple(solution)
                if solution_tuple not in node_mapping:
                    node_label = f"Solution {len(node_mapping) + 1}"
                    node_mapping[solution_tuple] = node_label
                    G.add_node(node_label, solution=solution, fitness=unique_fitnesses[i], iterations=solution_iterations[i], run_idx=run_idx, step=i)
                else:
                    node_label = node_mapping[solution_tuple]

                # Set start and end nodes for coloring later
                if i == 0:
                    start_nodes.add(node_label)
                if i == len(unique_solutions) - 1:
                    end_nodes.add(node_label)

            # Add edges based on transitions
            for j, (prev_solution, current_solution) in enumerate(transitions):
                prev_solution = tuple(prev_solution)
                current_solution = tuple(current_solution)
                if prev_solution in node_mapping and current_solution in node_mapping:
                    # transition = (node_mapping[prev_solution], node_mapping[current_solution])
                    # if transition not in transition_counts:
                    #     transition_counts[transition] = 0
                    # transition_counts[transition] += 1
                    G.add_edge(node_mapping[prev_solution], node_mapping[current_solution], weight=np.log10(solution_iterations[i]), color=edge_color)
    
    # Add trajectory nodes if provided
    if all_trajectories_list:
        # Determine optimisation goal
        optimisation_goal = determine_optimisation_goal(all_trajectories_list)

        # Add all sets of trajectories to the graph
        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            edge_color = algo_colors[idx % len(algo_colors)]  # Cycle through colors if there are more sets than colors

            selected_trajectories = []
            if n_runs_display > 0:
                selected_trajectories.extend(all_run_trajectories[:n_runs_display])
            if show_best:
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, optimisation_goal))
            if show_mean:
                selected_trajectories.extend([get_mean_run(all_run_trajectories)])
            if show_median:
                selected_trajectories.extend([get_median_run(all_run_trajectories)])
            if show_worst:
                anti_optimisation_goal = 'min' if optimisation_goal == 'max' else 'max'
                selected_trajectories.extend(select_top_runs_by_fitness(all_run_trajectories, 1, anti_optimisation_goal))

            add_trajectories_to_graph(selected_trajectories, edge_color)
        

        # Find the overall best solution across all sets of trajectories
        if optimisation_goal == "max":
            overall_best_fitness = max(
                max(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _ in all_run_trajectories
            )
        else:  # Minimisation
            overall_best_fitness = min(
                min(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _ in all_run_trajectories
            )
    
    # Add local optima nodes if provided
    if local_optima:
        # print(len(local_optima))
        local_optima_solutions, local_optima_fitnesses, local_optima_edges = local_optima
        for run_idx, (solutions, fitnesses, edges) in enumerate(zip(local_optima_solutions, local_optima_fitnesses, local_optima_edges)):
            # print(len(solutions))
            for i, solution in enumerate(solutions):
                # print(len(solution))
                solution_tuple = tuple(solution)
                if solution_tuple not in node_mapping:
                    node_label = f"Local Optimum {len(node_mapping) + 1}"
                    node_mapping[solution_tuple] = node_label
                    G.add_node(node_label, solution=solution, fitness=fitnesses[i])
            # print('nodes done')

            for prev_solution, current_solution in edges:
                prev_solution = tuple(prev_solution)
                current_solution = tuple(current_solution)
                if prev_solution in node_mapping and current_solution in node_mapping:
                    # Track transition frequency
                    transition = (node_mapping[prev_solution], node_mapping[current_solution])
                    if transition not in transition_counts:
                        transition_counts[transition] = 0
                    transition_counts[transition] += 1
                    # G.add_edge(node_mapping[prev_solution], node_mapping[current_solution], color='black')

    for (node1, node2), count in transition_counts.items():
        G.add_edge(node1, node2, weight=np.log10(count), color='black')

    # Find overall best solution from local optima
    if not all_trajectories_list:
        optimisation_goal = 'max'
    
    # if local_optima:
    #     local_optima_solutions, local_optima_fitnesses, local_optima_edges = local_optima
    #     if optimisation_goal == "max":
    #         local_optima_best_fitness = max(local_optima_fitnesses)
    #         if local_optima_best_fitness > overall_best_fitness:
    #             overall_best_fitness = local_optima_best_fitness
    #     else:  # Minimization
    #         local_optima_best_fitness = min(local_optima_fitnesses)
    #         if local_optima_best_fitness < overall_best_fitness:
    #             overall_best_fitness = local_optima_best_fitness

    # Find the overall best node
    for node, data in G.nodes(data=True):
        if data['fitness'] == overall_best_fitness:
            overall_best_node = node
            # print(overall_best_node)
            break

    # Prepare node colors
    node_colors = []
    for node in G.nodes():
        if node == overall_best_node:
            node_colors.append('red')
        elif node in start_nodes:
            node_colors.append('yellow')
        elif node in end_nodes:
            node_colors.append('grey')
        elif "Local Optimum" in node:
            node_colors.append(local_optima_color)
        else:
            # Check if the node exists in multiple sets of trajectories
            solution_tuple = next(key for key, value in node_mapping.items() if value == node)
            count_in_sets = 0
            for all_run_trajectories in all_trajectories_list:
                for unique_solutions, _, _, _ in all_run_trajectories:
                    if solution_tuple in set(tuple(sol) for sol in unique_solutions):
                        count_in_sets += 1
            if count_in_sets > 1:
                node_colors.append(node_color_shared)
            else:
                node_colors.append('skyblue')
    # print(f"Node Colors: {node_colors}")

    # Calculate node sizes
    if hide_nodes:
        # Node sizes set to zero (not shown)
        node_sizes = [0 for node in G.nodes()]
    elif use_solution_iterations:
        # Node sizes based on solution iterations
        node_sizes = [50 + G.nodes[node]['iterations'] * 20 for node in G.nodes()]
    else:
        # Node sizes based on the number of incoming edges (in-degree)
        node_sizes = [50 + G.in_degree(node) * 50 for node in G.nodes()]

    # Prepare node positions based on selected layout
    if layout == 'mds':
        # Use MDS to position nodes based on dissimilarity (Hamming distance)
        solutions = [data['solution'] for _, data in G.nodes(data=True)]
        n = len(solutions)
        dissimilarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dissimilarity_matrix[i][j] = hamming_distance(solutions[i], solutions[j])

        mds = MDS_sklearn(n_components=2, dissimilarity='precomputed', random_state=42)
        positions_2d = mds.fit_transform(dissimilarity_matrix)
        pos = {node: positions_2d[i] for i, node in enumerate(G.nodes())}
    elif layout == 'tsne':
        # Use t-SNE to position nodes based on dissimilarity (Hamming distance)
        solutions = [data['solution'] for _, data in G.nodes(data=True)]
        n = len(solutions)
        dissimilarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dissimilarity_matrix[i][j] = hamming_distance(solutions[i], solutions[j])

        # Initialize and fit t-SNE
        tsne = TSNE(n_components=2, metric='precomputed', random_state=42, init='random')
        positions_2d = tsne.fit_transform(dissimilarity_matrix)

        # Create a dictionary to map nodes to positions
        pos = {node: positions_2d[i] for i, node in enumerate(G.nodes())}
    elif layout == 'raw':
        # Directly use the 2D solution values as positions
        solutions = [data['solution'] for _, data in G.nodes(data=True)]
        pos = {node: solutions[i] for i, node in enumerate(G.nodes())}
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G, dim=2 if not plot_3D else 3)
    else:
        pos = nx.spring_layout(G, dim=2 if not plot_3D else 3)
    
    # create node_hover_text which holds node hover text information
    node_hover_text = []
    if hover_info_value == 'fitness':
        node_hover_text = [str(G.nodes[node]['fitness']) for node in G.nodes()]
    elif hover_info_value == 'iterations':
        node_hover_text = [str(G.nodes[node]['iterations']) for node in G.nodes()]
    elif hover_info_value == 'solutions':
        node_hover_text = [str(G.nodes[node]['solution']) for node in G.nodes()]

    # Prepare Plotly traces
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]][0], pos[edge[0]][1]
        x1, y1 = pos[edge[1]][0], pos[edge[1]][1]
        trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=edge[2].get('color', 'black')),
            hoverinfo='none'
        )
        edge_trace.append(trace)

    # Set x and y axis ranges based on slider
    if 'ENABLED' in use_range_slider:
        x_range = [-x_slider, x_slider]
        y_range = [-y_slider, y_slider]
    else:
        if len(G.nodes) > 0:
            x_values = [pos[node][0] for node in G.nodes()]
            y_values = [pos[node][1] for node in G.nodes()]
            x_range = [min(x_values) - 1, max(x_values) + 1]
            y_range = [min(y_values) - 1, max(y_values) + 1]
        else:
            x_range = [-10, 10]  # Default
            y_range = [-10, 10]  # Default

    if plot_3D:
        # 3D Plotting logic
        fig = go.Figure()
        for edge in G.edges(data=True):
            x0, y0, z0 = pos[edge[0]][0], pos[edge[0]][1], G.nodes[edge[0]]['fitness']
            x1, y1, z1 = pos[edge[1]][0], pos[edge[1]][1], G.nodes[edge[1]]['fitness']
            weight = edge[2]['weight']  # Access the edge weight
            trace = go.Scatter3d(
                x=[x0, x1],
                y=[y0, y1],
                z=[z0, z1],
                mode='lines',
                line=dict(width=weight * 1, color=edge[2].get('color', 'black')),
                hoverinfo='none'
            )
            fig.add_trace(trace)

        node_trace_3d = go.Scatter3d(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            z=[G.nodes[node]['fitness'] for node in G.nodes()],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                sizemode='area',  # Makes the node size scale with zoom
                size=node_sizes,
                color=node_colors,
                colorscale='YlGnBu',
                line_width=2
            ),
            text=node_hover_text,
            hoverinfo='text'
        )
        fig.add_trace(node_trace_3d)
        fig.update_layout(
            title="3D Trajectory Network Plot",
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(title='fitness')
            )
        )
    else:
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text' if show_labels else 'markers',
            hoverinfo='text',
            marker=dict(
                sizemode='area',  # Makes the node size scale with zoom
                showscale=False,
                colorscale='YlGnBu',
                size=node_sizes,
                color=node_colors,
                line_width=2
            )
        )

        for i, node in enumerate(G.nodes(data=True)):
            x, y = pos[node[0]]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node_hover_text[i]])

        fig = go.Figure(data=edge_trace + [node_trace],
                        layout=go.Layout(
                            title='Search Trajectory Network Plot',
                            width=700,
                            height=700,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
  
