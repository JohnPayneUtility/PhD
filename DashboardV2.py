import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.manifold import MDS as MDS_sklearn
from sklearn.manifold import TSNE

from DashboardHelpers import *

# ==========
# Data Loading and Formating
# ==========

# ---------- LON DATA ----------
# Load local optima
df_LONs = pd.read_pickle('results_LON.pkl')
print(f'df_LON columns: {df_LONs.columns}')
# print(df_LONs.head(10))
LON_hidden_cols = ['problem_name', 
                   'problem_type', 
                   'problem_goal', 
                   'dimensions', 
                   'opt_global',
                   'local_optima',
                   'fitness_values',
                   'edges'
                   ]
LON_display_columns = [col for col in df_LONs.columns if col not in LON_hidden_cols]

# ----------

# Load your data from the pickle file
df = pd.read_pickle('results.pkl')
df['opt_global'] = df['opt_global'].astype(float)
# print(df.columns)

df_no_lists = df.copy()
df_no_lists.drop([
    'unique_sols',
    'unique_fits',
    'noisy_fits',
    'sol_iterations',
    'sol_transitions',
], axis=1, inplace=True)

# Create subset DataFrames
display1_df = df.copy()
display1_df = display1_df[['problem_type',
                           'problem_goal',
                           'problem_name',
                           'dimensions',
                           'opt_global',
                           'PID']].drop_duplicates()

# For Table 2, group and aggregate data
display2_df = df[['problem_name',
                  'opt_global', 
                  'fit_func', 
                  'noise', 
                  'algo_type',
                  'algo_name', 
                  'n_unique_sols', 
                  'n_gens', 
                  'n_evals',
                  'final_fit',
                  'max_fit',
                  'min_fit']]
display2_df = df.groupby(
    ['problem_name', 'opt_global', 'fit_func', 'noise', 'algo_type']
).agg({
    'n_unique_sols': 'median',
    'n_gens': 'median',
    'n_evals': 'median',
    'final_fit': 'mean',
    'max_fit': 'max',
    'min_fit': 'min'
}).reset_index()
display2_hidden_cols = ['problem_type', 'problem_goal', 'problem_name', 'dimensions', 'opt_global']

display2_df = df.copy()
display2_df.drop([
    'n_gens',
    'n_evals',
    'stop_trigger',
    'n_unique_sols',
    'unique_sols',
    'unique_fits',
    'noisy_fits',
    'final_fit',
    'max_fit',
    'min_fit',
    'sol_iterations',
    'sol_transitions',
    'seed',
    'seed_signature'
], axis=1, inplace=True)
display2_hidden_cols = [
    'problem_type', 
    'problem_goal', 
    'problem_name', 
    'dimensions', 
    'opt_global',
    'PID'
    ]

# ==========
# Main Dashboard App
# ==========

app = dash.Dash(__name__, suppress_callback_exceptions=True)
# app = dash.Dash(__name__) # Don't suppress exceptions

# ---------- Style settings ----------
# Common style for unselected tabs
tab_style = {
    'height': '30px',
    'lineHeight': '30px',
    'fontSize': '14px',
    'padding': '0px'
}
# Common style for selected tabs
tab_selected_style = {
    'height': '30px',
    'lineHeight': '30px',
    'fontSize': '14px',
    'padding': '0px',
    'backgroundColor': '#ddd'  # for example
}

app.layout = html.Div([
    html.H2("LON/STN Dashboard", style={'textAlign': 'center'}),
    
    # Hidden stores to preserve selections from tables
    dcc.Store(id="table1-selected-store", data=[]),
    dcc.Store(id="table1tab2-selected-store", data=[]),
    dcc.Store(id="data-problem-specific", data=[]),
    dcc.Store(id="table2-selected-store", data=[]),

    # Hidden stores for plotting data
    dcc.Store(id="plot_2d_data", data=[]),
    dcc.Store(id="STN_data", data=[]),
    dcc.Store(id="LON_data", data=[]),
    dcc.Store(id="STN_data_processed", data=[]),
    dcc.Store(id="STN_series_labels", data=[]),
    dcc.Store(id="noisy_fitnesses_data", data=[]),
    
    # Tabbed section for problem selection
    html.Div([
        dcc.Tabs(id='problemTabSelection', value='p1', children=[
            dcc.Tab(label='Select problem', 
                    value='p1', style=tab_style, 
                    selected_style=tab_selected_style),
            dcc.Tab(label='Select additional problem (optional)', 
                    value='p2', 
                    style=tab_style, 
                    selected_style=tab_selected_style),
        ]),
        html.Div(id='problemTabsContent'),
    ], style={
        "border": "2px solid #ccc",
        "padding": "10px",
        "borderRadius": "5px",
        "margin": "10px",
    }),

    # Tabbed section for 2D performance plot
    html.Div([
        html.H3("2D Performance Plotting"),
        dcc.Tabs(id='2DPlotTabSelection', value='p1', children=[
            dcc.Tab(label='Line plot', 
                    value='p1', style=tab_style, 
                    selected_style=tab_selected_style),
            dcc.Tab(label='Box plot', 
                    value='p2', 
                    style=tab_style, 
                    selected_style=tab_selected_style),
            dcc.Tab(label='Data', 
                    value='p3', 
                    style=tab_style, 
                    selected_style=tab_selected_style),
        ]),
        html.Div(id='2DPlotTabContent'),
    ], style={
        "border": "2px solid #ccc",
        "padding": "10px",
        "borderRadius": "5px",
        "margin": "10px",
    }),
    
    # Table 2 is always visible and is filtered by the union of selections.
    html.H5("Table 2: Algorithms filtered by Selections"),
    dash_table.DataTable(
        id="table2",
        data=display2_df.to_dict("records"),  # initially full data
        columns=[{"name": col, "id": col} for col in display2_df.columns if col not in display2_hidden_cols],
        page_size=10,
        filter_action="native",
        sort_action='native',
        row_selectable="multi",
        selected_rows=[],
        style_table={"overflowX": "auto"},
    ),
    html.Div(id="table2-selected-output", style={
        "margin": "10px", "padding": "10px", "border": "1px solid #ccc"
    }),
    # html.Hr(),

    # Tabbed section for 3D LON/STN plot
    html.Div([
        html.H3("STN/LON Plot"),
        dcc.Tabs(id='STNPlotTabSelection', value='p1', children=[
            dcc.Tab(label='2D STN', 
                    value='p1', style=tab_style, 
                    selected_style=tab_selected_style),
            dcc.Tab(label='3D STN', 
                    value='p2', 
                    style=tab_style, 
                    selected_style=tab_selected_style),
            dcc.Tab(label='3D Joint STN LON', 
                    value='p3', 
                    style=tab_style, 
                    selected_style=tab_selected_style),
        ]),
        html.Div(id='STNPlotTabContent'),
    ], style={
        "border": "2px solid #ccc",
        "padding": "10px",
        "borderRadius": "5px",
        "margin": "10px",
    }),
    # PORTED FROM V1
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
            {'label': 'Hide STN Nodes', 'value': 'hide_STN_nodes'},
            {'label': 'Hide LON Nodes', 'value': 'hide_LON_nodes'},
            {'label': '3D Plot', 'value': 'plot_3D'},
            {'label': 'Use Solution Iterations', 'value': 'use_solution_iterations'},
            {'label': 'Use strength for LON node size', 'value': 'LON_node_strength'},
            {'label': 'Colour LON by fitness', 'value': 'local_optima_color'}
        ],
        value=['plot_3D', 'LON_node_strength']  # Set default values
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
    html.Label("Node size scaling:"),
    dcc.Slider(
    id='node-size-slider',
    min=1,
    max=100,
    step=1,
    value=50,  # Default scaling factor
    marks={i: str(i) for i in range(1, 101, 10)},
    tooltip={"placement": "bottom", "always_visible": False}
    ),
    html.Label("LON Edge thickness:"),
    dcc.Slider(
    id='LON-edge-size-slider',
    min=1,
    max=100,
    step=1,
    value=5,  # Default scaling factor
    marks={i: str(i) for i in range(1, 100, 10)},
    tooltip={"placement": "bottom", "always_visible": False}
    ),
    # Replaced with input box
    # html.Label("STN Edge Thickness:"),
    # dcc.Slider(
    # id='STN-edge-size-slider',
    # min=1,
    # max=100,
    # step=1,
    # value=50,  # Default scaling factor
    # marks={i: str(i) for i in range(1, 101, 10)},
    # tooltip={"placement": "bottom", "always_visible": False}
    # ),
    html.Div([
        html.Label("STN Edge Thickness"),
    ], style={'display': 'inline-block', 'margin-right': '10px'}),
    html.Div([
        dcc.Input(
            id='STN-edge-size-slider',
            type='number',
            min=0,
            max=100,
            step=1,
            value=5
        ),
    ], style={'display': 'inline-block'}),

    html.Label("Local Optima Node Opacity:"),
    dcc.Slider(
        id='local-optima-node-opacity-slider',
        min=0.0,
        max=1.0,
        step=0.1,
        value=1.0,
        marks={i/10: f"{i/10:.1f}" for i in range(1, 11)},
    ),
    html.Label("Local Optima Edge Opacity:"),
    dcc.Slider(
        id='local-optima-edge-opacity-slider',
        min=0.0,
        max=1.0,
        step=0.1,
        value=1.0,
        marks={i/10: f"{i/10:.1f}" for i in range(1, 11)},
    ),
    html.Label("Other Node Opacity:"),
    dcc.Slider(
        id='STN-node-opacity-slider',
        min=0.0,
        max=1.0,
        step=0.1,
        value=1.0,
        marks={i/10: f"{i/10:.1f}" for i in range(1, 11)},
    ),
    html.Label("Other Edge Opacity:"),
    dcc.Slider(
        id='STN-edge-opacity-slider',
        min=0.0,
        max=1.0,
        step=0.1,
        value=1.0,
        marks={i/10: f"{i/10:.1f}" for i in range(1, 11)},
    ),
    dcc.Graph(id='trajectory-plot'),
    html.Div(id="print_STN_series_labels", style={
        "margin": "10px", "padding": "10px", "border": "1px solid #ccc"
    }),
])

# ------------------------------
# Callback: Render Problem Tab Content
# ------------------------------

@app.callback(
    Output('problemTabsContent', 'children'),
    [Input('problemTabSelection', 'value'),
     Input('table1-selected-store', 'data'),
     Input('table1tab2-selected-store', 'data')]
)
def render_content_problem_tab(tab, stored_selection_tab1, stored_selection_tab2):
    if tab == 'p1':
        return html.Div([
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "padding": "10px",
                    "marginTop": "0px"
                },
                children=[
                    dash_table.DataTable(
                        id="table1",
                        data=display1_df.to_dict("records"),
                        columns=[{"name": col, "id": col} for col in display1_df.columns],
                        page_size=10,
                        filter_action="native",
                        row_selectable="single",
                        # Use stored selection from Tab 1
                        selected_rows=stored_selection_tab1 if stored_selection_tab1 else [],
                        style_table={"overflowX": "auto"},
                    )
                ]
            ),
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "flex-start",
                    "alignItems": "center",
                    "padding": "10px",
                    "marginTop": "0px"
                },
                children=[
                    dash_table.DataTable(
                        id="LON_table",
                        data=df_LONs[LON_display_columns].to_dict("records"),
                        columns=[{"name": col, "id": col} for col in LON_display_columns],
                        page_size=10,
                        # filter_action="native",
                        row_selectable="single",
                        style_table={"overflowX": "auto"},
                    )
                ]
            )
        ])
    elif tab == 'p2':
        # ADD ANOTHER VERSION OF TAB 1 HERE WITH UPDATED NAMES
        return html.Div([
            html.H3('Content for Tab Two'),
            dash_table.DataTable(
                id="table1_tab2",
                data=display1_df.to_dict("records"),
                columns=[{"name": col, "id": col} for col in display1_df.columns],
                page_size=10,
                filter_action="native",
                row_selectable="single",
                selected_rows=stored_selection_tab2 if stored_selection_tab2 else [],
                style_table={"overflowX": "auto"},
            )
        ])

# ------------------------------
# Callbacks: Update Selection Stores
# ------------------------------

@app.callback(
    Output("table1-selected-store", "data"),
    Input("table1", "selected_rows"),
    prevent_initial_call=True
)
def update_table1_store(selected_rows):
    return selected_rows

@app.callback(
    Output("table1tab2-selected-store", "data"),
    Input("table1_tab2", "selected_rows"),
    prevent_initial_call=True
)
def update_table1tab2_store(selected_rows):
    return selected_rows

@app.callback(
    Output("table2-selected-store", "data"),
    Input("table2", "selected_rows"),
    prevent_initial_call=True
)
def update_table2_store(selected_rows):
    return selected_rows

# ------------------------------
# Callback: Filter Table 2 Based on Selections
# ------------------------------

# Update data store filtered by specific problem
@app.callback(
    Output("data-problem-specific", "data"),
    [Input("table1-selected-store", "data"),
     Input("table1tab2-selected-store", "data")]
)
def filter_table2(selection1, selection2):
    union = set()
    # For Table 1 (Tab 1), use display1_df to retrieve problem_name.
    if selection1:
        for idx in selection1:
            if idx < len(display1_df):
                union.add(display1_df.iloc[idx]['PID'])
    # For Table 1 on Tab 2, also use display1_df.
    if selection2:
        for idx in selection2:
            if idx < len(display1_df):
                union.add(display1_df.iloc[idx]['PID'])
    if not union:
        return display2_df.to_dict("records")
    else:
        filtered_df = display2_df[display2_df['PID'].isin(union)]
        return filtered_df.to_dict("records")

# Update table 2 to use problem specific data store
@app.callback(
    Output("table2", "data"),
    Input("data-problem-specific", "data")
)
def update_table2(data):
    if data is None:
        return []
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    return df.to_dict('records')

# ------------------------------
# Callback: Display Selected Rows from Table 2
# ------------------------------

@app.callback(
    Output("table2-selected-output", "children"),
    Input("table2", "selected_rows"),
    State("table2", "data")
)
def update_table2_selected(selected_rows, table2_data):
    if not selected_rows:
        return "No rows selected in Table 2."
    selected_data = [table2_data[i] for i in selected_rows]
    return f"Table 2 selected rows: {selected_data}"

# ------------------------------
# 2D Plot
# ------------------------------
# ---------- Generate data for 2D performance plot by filtering main data with table 2 selection ----------
filter_columns = [col for col in display2_df.columns]
@app.callback(
    Output('plot_2d_data', 'data'),
    Input('table2', 'derived_virtual_data')
)
def update_filtered_view(filtered_data):
    # If no filtering is applied, return the full data.
    if not filtered_data:
        return df_no_lists.to_dict('records')
    
    # Convert the filtered data to a DataFrame.
    df_filtered = pd.DataFrame(filtered_data)
    # print("df_filtered:")
    # print(df_filtered.head())
    
    mask = pd.Series(True, index=df_no_lists.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            # print(f"Filtering column {col} with allowed values: {allowed_values}")
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df_no_lists[col].isin(allowed_values) | df_no_lists[col].isnull())
            else:
                mask &= df_no_lists[col].isin(allowed_values)
    
    df_result = df_no_lists[mask]
    # print("Filtered result (first few rows):")
    # print(df_result.head())
    return df_result.to_dict('records')

# ---------- 2D plot callbacks ----------
# Plot data table
@app.callback(
    Output('plot_2d_data_table', 'data'),
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    return data

# 2D line plot
@app.callback(
    Output('2DLinePlot', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_line(plot_df)
    return plot
# 2D bar plot
@app.callback(
    Output('2DBoxPlot', 'figure'),
    Input('plot_2d_data', 'data')
)
def display_stored_data(data):
    plot_df = pd.DataFrame(data)
    plot = plot2d_box(plot_df)
    return plot

# ---------- Render 2D plot content in tabbed view ----------
@app.callback(
    Output('2DPlotTabContent', 'children'),
    Input('2DPlotTabSelection', 'value')
)
def render_content_2DPlot_tab(tab):
    if tab == 'p1':
        return html.Div([
            dcc.Graph(id='2DLinePlot'),
        ])
    elif tab == 'p2':
        return html.Div([
            dcc.Graph(id='2DBoxPlot'),
        ])
    elif tab == 'p3':
        return html.Div([
            dash_table.DataTable(
                id='plot_2d_data_table',
                columns=[{'name': col, 'id': col} for col in df_no_lists.columns],
                page_size=10,
                data=[],
                style_table={
                    'maxWidth': '100%',  # limits the table to the width of the container
                    'overflowX': 'auto'  # adds a scrollbar if needed
                },
            )
        ])
# ------------------------------
# LON Plots
# ------------------------------

# Filter LON dataframe by selected problem using table 1 selection
@app.callback(
    Output('LON_data', 'data'),
    Input("LON_table", "selected_rows"),
    State("LON_table", "data")
)
def update_filtered_view(selected_rows, LON_table_data):
    # Check if no selection has been made and if so return empty dataframe to store
    # print(f'table 2 selected_rows: {selected_rows}, length: {len(selected_rows)}')
    if len(selected_rows) == 0:
        blank_df = pd.DataFrame(columns=df_LONs.columns)
        return blank_df.to_dict('records')

    selected_data = [LON_table_data[i] for i in selected_rows]
    # print(selected_data)

    LON_table_filter_cols = ['PID', 'LON_Algo', 'n_flips_mut', 'n_flips_pert', 'compression_val', 'n_local_optima']
    filter_columns = [col for col in LON_display_columns]
    
    # Convert the filtered data to a DataFrame.
    df_filtered = pd.DataFrame(selected_data)
    # print("df_filtered:")
    # print(df_filtered.head())
    
    mask = pd.Series(True, index=df_LONs.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            # print(f"Filtering column {col} with allowed values: {allowed_values}")
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df_LONs[col].isin(allowed_values) | df_LONs[col].isnull())
            else:
                mask &= df_LONs[col].isin(allowed_values)
    
    df_result = df_LONs[mask]
    # print("Filtered result (first few rows):")
    # print(df_result['edges'])
    # print(df_result.columns)
    LON_plotting_cols = ['local_optima', 'fitness_values', 'edges']
    df_result = df_result.loc[:, LON_plotting_cols]
    dict_result = df_result.to_dict('records')
    # dict_result_SE = convert_to_split_edges_format(dict_result)

    combined_dict = {
    "local_optima": [],
    "fitness_values": [],
    "edges": {}
    }

    for row in dict_result:
        combined_dict["local_optima"].extend(row["local_optima"])
        combined_dict["fitness_values"].extend(row["fitness_values"])

        for (source, target), weight in row["edges"].items():
            if (source, target) in combined_dict["edges"]:
                combined_dict["edges"][(source, target)] += weight  # Accumulate edge weights
            else:
                combined_dict["edges"][(source, target)] = weight  # Add new edge

    # Now pass the **merged** dictionary to convert_to_split_edges_format()
    dict_result_SE = convert_to_split_edges_format(combined_dict)

    return dict_result_SE

# Filter main dataframe for STN data using table 2 selection
@app.callback(
    Output('STN_data', 'data'),
    Input("table2", "selected_rows"),
    State("table2", "data")
)
def update_filtered_view(selected_rows, table2_data):
    # Check if no selection has been made and if so return empty dataframe to store
    # print(f'table 2 selected_rows: {selected_rows}, length: {len(selected_rows)}')
    if len(selected_rows) == 0:
        blank_df = pd.DataFrame(columns=df.columns)
        return blank_df.to_dict('records')

    selected_data = [table2_data[i] for i in selected_rows]
    # print(selected_data)

    filter_columns = [col for col in display2_df.columns]
    
    # Convert the filtered data to a DataFrame.
    df_filtered = pd.DataFrame(selected_data)
    # print("df_filtered:")
    # print(df_filtered.head())
    
    mask = pd.Series(True, index=df.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            # print(f"Filtering column {col} with allowed values: {allowed_values}")
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df[col].isin(allowed_values) | df[col].isnull())
            else:
                mask &= df[col].isin(allowed_values)
    
    df_result = df[mask]
    # print("Filtered result (first few rows):")
    # print(df_result.head())
    # print(df_result.columns)
    return df_result.to_dict('records')

@app.callback(
    [Output('STN_data_processed', 'data'),
     Output('STN_series_labels', 'data'),
     Output('noisy_fitnesses_data', 'data')],
    Input('STN_data', 'data'),
)
def process_STN_data(df, group_cols=['algo_name', 'noise']):
    df = pd.DataFrame(df)
    STN_data = []
    STN_series = []
    Noise_data = []
    
    # Group by multiple columns
    grouped = df.groupby(group_cols)
    
    for group_key, group_df in grouped:
        runs = []

        for _, row in group_df.iterrows():
            run = [
                row['unique_sols'],
                row['unique_fits'],
                row['noisy_fits'],
                row['sol_iterations'],
                row['sol_transitions']
            ]
            runs.append(run)
        
        # Use tuple(group_key) as dictionary keys for clear indexing
        STN_data.append(runs)
        STN_series.append(group_key)
    
    return STN_data, STN_series, Noise_data

@app.callback(
    Output('print_STN_series_labels', "children"),
    Input('STN_series_labels', 'data')
)
def update_table2_selected(series_list):
    if not series_list:
        return "No rows selected in Table 2."
    # series_labels = [series_list[i] for i in series_list]
    return f"Plotted series: {series_list}"

# ==========
# plot
# ==========
algo_colors = ['blue', 'orange', 'purple', 'brown', 'cyan', 'magenta']
@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('options', 'value'),
     Input('run-options', 'value'),
     Input('layout', 'value'),
     Input('hover-info', 'value'),
     Input('STN_data_processed', 'data'),
     Input('run-selector', 'value'),
     Input('LON_data', 'data'),
     Input('use-range-sliders', 'value'),
     Input('x-axis-slider', 'value'),
     Input('y-axis-slider', 'value'),
     Input('node-size-slider', 'value'),
     Input('LON-edge-size-slider', 'value'),
     Input('STN-edge-size-slider', 'value'),
     Input('local-optima-node-opacity-slider', 'value'),
     Input('local-optima-edge-opacity-slider', 'value'),
     Input('STN-node-opacity-slider', 'value'),
     Input('STN-edge-opacity-slider', 'value'),
     Input('noisy_fitnesses_data', 'data')]
)
def update_plot(options, run_options, layout_value, hover_info_value, all_trajectories_list, n_runs_display, local_optima, use_range_slider, x_slider, y_slider, node_size_slider, LON_edge_size_slider, STN_edge_size_slider, LON_node_opac, LON_edge_opac, STN_node_opac, STN_edge_opac, noisy_fitnesses_list):
    # Options from checkboxes
    show_labels = 'show_labels' in options
    hide_STN_nodes = 'hide_STN_nodes' in options
    hide_LON_nodes = 'hide_LON_nodes' in options
    plot_3D = 'plot_3D' in options
    use_solution_iterations = 'use_solution_iterations' in options
    LON_node_strength = 'LON_node_strength' in options
    local_optima_color = 'local_optima_color' in options

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

    # Add nodes and edges for each set of trajectories
    node_mapping = {}  # To ensure unique solutions map to the same node
    start_nodes = set()
    end_nodes = set()
    overall_best_fitness = 0
    overall_best_node = None
    
    # # Function to add nodes and edges to the graph
    # def add_trajectories_to_graph(all_run_trajectories, edge_color):
    #     for run_idx, (unique_solutions, unique_fitnesses, _, solution_iterations, transitions) in enumerate(all_run_trajectories):
    #         for i, solution in enumerate(unique_solutions):
    #             solution_tuple = tuple(solution)
    #             if solution_tuple not in node_mapping:
    #                 node_label = f"Solution {len(node_mapping) + 1}"
    #                 node_mapping[solution_tuple] = node_label
    #                 G.add_node(node_label, solution=solution, fitness=unique_fitnesses[i], iterations=solution_iterations[i], run_idx=run_idx, step=i)
    #             else:
    #                 node_label = node_mapping[solution_tuple]

    #             # Set start and end nodes for coloring later
    #             if i == 0:
    #                 start_nodes.add(node_label)
    #             if i == len(unique_solutions) - 1:
    #                 end_nodes.add(node_label)

    #         # Add edges based on transitions
    #         for j, (prev_solution, current_solution) in enumerate(transitions):
    #             prev_solution = tuple(prev_solution)
    #             current_solution = tuple(current_solution)
    #             if prev_solution in node_mapping and current_solution in node_mapping:
    #                 G.add_edge(node_mapping[prev_solution], node_mapping[current_solution], weight=STN_edge_size_slider, color=edge_color)
    
    def add_trajectories_to_graph(all_run_trajectories, edge_color):
        for run_idx, entry in enumerate(all_run_trajectories):
            if len(entry) != 5:
                print(f"Skipping malformed entry {entry}, expected 5 elements but got {len(entry)}")
                continue
            
            unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions = entry

            if any(x is None for x in (unique_solutions, unique_fitnesses, noisy_fitnesses, solution_iterations, transitions)):
                print(f"Skipping run {run_idx} due to None values: {entry}")
                continue

            print(unique_fitnesses)
            print(noisy_fitnesses)
            noisy_fitnesses = [int(fit) for fit in noisy_fitnesses]
            print(noisy_fitnesses)
            for i, solution in enumerate(unique_solutions):
                solution_tuple = tuple(solution)
                if solution_tuple not in node_mapping:
                    node_label = f"Solution {len(node_mapping) + 1}"
                    node_mapping[solution_tuple] = node_label
                    G.add_node(node_label, solution=solution, fitness=unique_fitnesses[i], iterations=solution_iterations[i], run_idx=run_idx, step=i)
                else:
                    node_label = node_mapping[solution_tuple]

                # Add noisy fitness node
                noisy_node_label = f"Noisy {node_label}"
                if noisy_node_label not in G.nodes:
                    print(f"Existing nodes: {list(G.nodes())}")
                    print(f"Adding node: {noisy_node_label} with fitness {noisy_fitnesses[i]}")
                    try:
                        G.add_node(noisy_node_label, solution=solution, fitness=noisy_fitnesses[i])
                    except Exception as e:
                        print(f"Error adding noisy node: {noisy_node_label}, {e}")
                    G.add_edge(node_label, noisy_node_label, weight=STN_edge_size_slider, color='gray', style='dotted')

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
                    G.add_edge(node_mapping[prev_solution], node_mapping[current_solution], weight=STN_edge_size_slider, color=edge_color)

    # Add trajectory nodes if provided
    if all_trajectories_list:
        # Determine optimisation goal
        
        optimisation_goal = determine_optimisation_goal(all_trajectories_list)

        # Add all sets of trajectories to the graph
        # print(f"Checking all_trajectories_list: {all_trajectories_list}")
        for idx, all_run_trajectories in enumerate(all_trajectories_list):
            # print(f"Checking all_run_trajectories: {all_run_trajectories}")
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
                max(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _, _ in all_run_trajectories
            )
        else:  # Minimisation
            overall_best_fitness = min(
                min(best_fitnesses) for all_run_trajectories in all_trajectories_list for _, best_fitnesses, _, _, _ in all_run_trajectories
            )
    
    # Add local optima nodes if provided
    def filter_negative_LO(local_optima):
        # Create filtered lists for nodes and fitness values.
        filtered_nodes = []
        filtered_fitness_values = []
        
        # Loop through nodes and their fitness values, keeping only non-negative ones.
        for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
            if fitness >= 0:
                filtered_nodes.append(opt)
                filtered_fitness_values.append(fitness)
        
        # Create a set of allowed node tuples for filtering edges.
        allowed_nodes = {tuple(opt) for opt in filtered_nodes}
        
        # Filter the edges so that both source and target are in the allowed set.
        filtered_edges = {}
        for (source, target), weight in local_optima["edges"].items():
            if tuple(source) in allowed_nodes and tuple(target) in allowed_nodes:
                filtered_edges[(source, target)] = weight

        # Create a new dataset with the filtered data.
        filtered_local_optima = {
            "local_optima": filtered_nodes,
            "fitness_values": filtered_fitness_values,
            "edges": filtered_edges
        }
        return filtered_local_optima
    
    if local_optima:
        local_optima = convert_to_single_edges_format(local_optima)
        # local_optima = pd.DataFrame(local_optima).apply(convert_to_single_edges_format, axis=1)
        local_optima = filter_negative_LO(local_optima)
        print(len(local_optima["local_optima"]))

        max_weight = max(local_optima["edges"].values(), default=1)

        for opt, fitness in zip(local_optima["local_optima"], local_optima["fitness_values"]):
            solution_tuple = tuple(opt)  # Ensure the solution is a tuple
            if solution_tuple not in node_mapping:
                node_label = f"Local Optimum {len(node_mapping) + 1}"
                node_mapping[solution_tuple] = node_label
                G.add_node(node_label, solution=opt, fitness=fitness)

        # Add edges to the graph directly from the `edges` dictionary
        for (source, target), weight in local_optima["edges"].items():
            source_tuple = tuple(source)
            target_tuple = tuple(target)

            # Ensure nodes exist in the mapping
            if source_tuple in node_mapping and target_tuple in node_mapping:
                source_label = node_mapping[source_tuple]
                target_label = node_mapping[target_tuple]

                # Add the edge with the weight from the 
                normalized_weight = weight / max_weight
                size = normalized_weight * LON_edge_size_slider
                G.add_edge(source_label, target_label, weight=size, color='black')

    # Find overall best solution from local optima
    if not all_trajectories_list:
        optimisation_goal = 'max'
    
    # Calculate local optima fitness ranges
    local_optima_nodes = [node for node in G.nodes() if "Local Optimum" in node]
    if local_optima_nodes:
        local_optima_fitnesses = [G.nodes[node]['fitness'] for node in local_optima_nodes]
        if optimisation_goal == "max":
            min_lo_fitness = min(local_optima_fitnesses)
            max_lo_fitness = max(local_optima_fitnesses)
            if max_lo_fitness > overall_best_fitness:
                overall_best_fitness = max_lo_fitness
        else:
            # If optimisation goal is minimisation
            min_lo_fitness = max(local_optima_fitnesses)
            max_lo_fitness = min(local_optima_fitnesses)
            if max_lo_fitness < overall_best_fitness:
                overall_best_fitness = max_lo_fitness
    else:
        # Fallback values if no local optima exist (won't be used if there are none)
        min_lo_fitness = max_lo_fitness = None

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
            node_colors.append(convert_to_rgba('red'))
        elif node in start_nodes:
            node_colors.append(convert_to_rgba('yellow'))
        elif node in end_nodes:
            node_colors.append(convert_to_rgba('grey'))
        elif "Local Optimum" in node:
            if local_optima_color:
                fitness = G.nodes[node]['fitness']
                node_colors.append(fitness_to_color(fitness, min_lo_fitness, max_lo_fitness, LON_node_opac))
            else:
                node_colors.append(convert_to_rgba('black', LON_node_opac))
        else:
            # Check if the node exists in multiple sets of trajectories
            solution_tuple = next((key for key, value in node_mapping.items() if value == node), None)
            if solution_tuple is None:
                # Handle the missing mapping appropriately, e.g. skip this node or log an error.
                continue
            count_in_sets = 0
            for all_run_trajectories in all_trajectories_list:
                for unique_solutions, _, _, _, _ in all_run_trajectories:
                    if solution_tuple in set(tuple(sol) for sol in unique_solutions):
                        count_in_sets += 1
            if count_in_sets > 1:
                node_colors.append(convert_to_rgba(node_color_shared))
            else:
                node_colors.append(convert_to_rgba('skyblue', STN_node_opac))
    # print(f"Node Colors: {node_colors}")

    # Calculate node sizes
    node_sizes = []
    for node in G.nodes():
        if hide_LON_nodes and "Local Optimum" in node:
            node_sizes.append(0)
        elif LON_node_strength and "Local Optimum" in node:
            incoming_edges = G.in_edges(node, data=True)
            local_optimum_size = sum(edge_data.get('weight', 0) for _, _, edge_data in incoming_edges)
            node_sizes.append(50 + local_optimum_size * node_size_slider)
        elif hide_STN_nodes and "Local Optimum" not in node:
            node_sizes.append(0)
        elif use_solution_iterations and "Local Optimum" not in node:
            node_sizes = [node_size_slider + G.nodes[node]['iterations'] * node_size_slider for node in G.nodes()]
        else:
            # Node sizes based on the number of incoming edges (in-degree)
            node_sizes = [node_size_slider + G.in_degree(node) * node_size_slider for node in G.nodes()]

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
                line=dict(width=weight * 1, color=convert_to_rgba(edge[2].get('color', 'black'), STN_edge_opac)),
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

# ==========
# RUN
# ==========

if __name__ == '__main__':
    app.run_server(debug=True)