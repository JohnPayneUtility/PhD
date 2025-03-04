import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd

# Load your data from the pickle file
df = pd.read_pickle('results.pkl')
df['opt_global'] = df['opt_global'].astype(float)

# Create subset DataFrames
display1_df = df[['problem_type',
                  'problem_goal',
                  'problem_name',
                  'dimensions',
                  'opt_global']].drop_duplicates()

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
    ['problem_name', 'opt_global', 'fit_func', 'noise', 'algo_name']
).agg({
    'n_unique_sols': 'median',
    'n_gens': 'median',
    'n_evals': 'median',
    'final_fit': 'mean',
    'max_fit': 'max',
    'min_fit': 'min'
}).reset_index()

# Optionally, columns to hide in Table 2 display (if any)
display2_hidden_cols = ['problem_type', 'problem_goal', 'problem_name', 'dimensions', 'opt_global']

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("LON/STN Dashboard", style={'textAlign': 'center'}),
    
    # Hidden stores to preserve selections from Table 1 (Tab 1), Table 1 on Tab 2, and Table 2
    dcc.Store(id="table1-selected-store", data=[]),
    dcc.Store(id="table1tab2-selected-store", data=[]),
    dcc.Store(id="table2-selected-store", data=[]),
    
    # Tabbed section for problem selection
    html.Div([
        dcc.Tabs(id='problemTabSelection', value='p1', children=[
            dcc.Tab(label='Select problem', value='p1'),
            dcc.Tab(label='Select additional problem (optional)', value='p2'),
        ]),
        html.Div(id='problemTabsContent'),
    ], style={
        "border": "2px solid #ccc",
        "padding": "10px",
        "borderRadius": "5px",
        "margin": "10px"
    }),
    
    # Table 2 is always visible and is filtered by the union of selections.
    html.H5("Table 2: Algorithms filtered by Selections"),
    dash_table.DataTable(
        id="table2",
        data=display2_df.to_dict("records"),  # initially full data
        columns=[{"name": col, "id": col} for col in display2_df.columns if col not in display2_hidden_cols],
        page_size=10,
        filter_action="native",
        row_selectable="multi",
        selected_rows=[],
        style_table={"overflowX": "auto"},
    ),
    html.Div(id="table2-selected-output", style={
        "margin": "10px", "padding": "10px", "border": "1px solid #ccc"
    })
])

# ------------------------------
# Callback: Render Tab Content
# ------------------------------

@app.callback(
    Output('problemTabsContent', 'children'),
    [Input('problemTabSelection', 'value'),
     Input('table1-selected-store', 'data'),
     Input('table1tab2-selected-store', 'data')]
)
def render_content(tab, stored_selection_tab1, stored_selection_tab2):
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
                        id="ProbTypeDropDown",
                        options=[{'label': val, 'value': val} 
                                 for val in display1_df['problem_type'].dropna().unique()],
                        placeholder="Problem type",
                        style={"width": "200px", "marginRight": "5px"}
                    ),
                    dcc.Dropdown(
                        id="dropdown2",
                        options=[{"label": "Option X", "value": "X"},
                                 {"label": "Option Y", "value": "Y"}],
                        value="X",
                        style={"width": "200px", "marginRight": "5px"}
                    ),
                    dcc.Dropdown(
                        id="dropdown3",
                        options=[{"label": "Option 1", "value": 1},
                                 {"label": "Option 2", "value": 2}],
                        value=1,
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

@app.callback(
    Output("table2", "data"),
    [Input("table1-selected-store", "data"),
     Input("table1tab2-selected-store", "data"),
     Input("table2-selected-store", "data")]
)
def filter_table2(selection1, selection2, selection3):
    union = set()
    # For Table 1 (Tab 1), use display1_df to retrieve problem_name.
    if selection1:
        for idx in selection1:
            if idx < len(display1_df):
                union.add(display1_df.iloc[idx]['problem_name'])
    # For Table 1 on Tab 2, also use display1_df.
    if selection2:
        for idx in selection2:
            if idx < len(display1_df):
                union.add(display1_df.iloc[idx]['problem_name'])
    # For Table 2 selection, use display2_df.
    if selection3:
        table2_full = display2_df.to_dict('records')
        for idx in selection3:
            if idx < len(display2_df):
                union.add(display2_df.iloc[idx]['problem_name'])
    if not union:
        return display2_df.to_dict("records")
    else:
        filtered_df = display2_df[display2_df['problem_name'].isin(union)]
        return filtered_df.to_dict("records")

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

if __name__ == '__main__':
    app.run_server(debug=True)
