import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from DashboardHelpers import *

# ==========
# Data Loading and Formating
# ==========

# Load your data from the pickle file
df = pd.read_pickle('results.pkl')
df['opt_global'] = df['opt_global'].astype(float)
print(df.columns)

df_no_lists = df.copy()
df_no_lists.drop([
    'unique_sols',
    'unique_fits',
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
    html.Hr(),

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

    # dash_table.DataTable(
    #     id='plot_2d_data_table',
    #     columns=[{'name': col, 'id': col} for col in df_no_lists.columns],
    #     page_size=10,
    #     data=[]  # Initially empty, will be updated via callback
    # ),
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
                    "marginBottom": "0px",
                    "marginLeft": "30px"
                },
                children=[
                    dcc.Dropdown(
                        id="tab1check1",
                        options=[{'label': val, 'value': val} 
                                 for val in display1_df['problem_type'].dropna().unique()],
                        placeholder="Problem type",
                        style={"width": "200px", "marginRight": "5px"}
                    ),
                    dcc.Dropdown(
                        id="tab1check2",
                        options=[{"label": "Option X", "value": "X"},
                                 {"label": "Option Y", "value": "Y"}],
                        value="P",
                        style={"width": "200px", "marginRight": "5px"}
                    ),
                    dcc.Dropdown(
                        id="tab1check3",
                        options=[{"label": "Option 1", "value": 1},
                                 {"label": "Option 2", "value": 2}],
                        value="P",
                        style={"width": "200px", "marginRight": "5px"}
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
# Callback: 2D plot data
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
    print("df_filtered:")
    print(df_filtered.head())
    
    mask = pd.Series(True, index=df_no_lists.index)
    for col in filter_columns:
        if col in df_filtered.columns:
            allowed_values = df_filtered[col].unique()
            print(f"Filtering column {col} with allowed values: {allowed_values}")
            # If any allowed value is null (None or np.nan), allow null rows.
            if any(pd.isnull(allowed_values)):
                mask &= (df_no_lists[col].isin(allowed_values) | df_no_lists[col].isnull())
            else:
                mask &= df_no_lists[col].isin(allowed_values)
    
    df_result = df_no_lists[mask]
    print("Filtered result (first few rows):")
    print(df_result.head())
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

# ==========
# RUN
# ==========

if __name__ == '__main__':
    app.run_server(debug=True)