import dash
from dash import html
from dash import Input, Output
import dash_tabulator as dt
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "problem_name": ["p1", "p1", "p1", "p1", "p2", "p2"],
    "opt_global": [1, 1, 1, 1, 2, 2],
    "fit_func": ["f1", "f1", "f1", "f1", "f2", "f2"],
    "noise": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
    "algo_type": ["TypeA", "TypeA", "TypeB", "TypeB", "TypeA", "TypeA"],
    "algo_name": ["algo1", "algo2", "algo3", "algo4", "algo1", "algo2"],
    "n_unique_sols": [10, 20, 30, 40, 15, 25],
    "n_gens": [100, 200, 300, 400, 150, 250],
    "n_evals": [1000, 2000, 3000, 4000, 1500, 2500],
    "final_fit": [0.9, 0.8, 0.85, 0.7, 0.95, 0.88],
    "max_fit": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "min_fit": [0.5, 0.6, 0.55, 0.65, 0.52, 0.62]
})

# Parent grouping columns
parent_cols = ["problem_name", "opt_global", "fit_func", "noise", "algo_type"]

nested_data = []

# Group the DataFrame by the parent columns
for parent_key, group in df.groupby(parent_cols):
    # Create a parent dict from the parent grouping keys
    parent_data = dict(zip(parent_cols, parent_key))
    
    # Collect the unique algos in this group for the parent row as a list
    parent_algos = sorted(group["algo_name"].unique())
    parent_data["algo_names_list"] = parent_algos  # store as a list, not a string
    
    # Prepare the child rows
    children = []
    for algo_name, sub_group in group.groupby("algo_name"):
        child = dict(parent_data)  # copy parent data so child inherits filtering fields
        # For the child's algo_names_list, store just this one algo
        child["algo_names_list"] = [algo_name]
        # Aggregate metrics
        child["n_unique_sols"] = sub_group["n_unique_sols"].median()
        child["n_gens"] = sub_group["n_gens"].median()
        child["n_evals"] = sub_group["n_evals"].median()
        child["final_fit"] = sub_group["final_fit"].mean()
        child["max_fit"] = sub_group["max_fit"].max()
        child["min_fit"] = sub_group["min_fit"].min()
        children.append(child)
    
    parent_data["children"] = children
    nested_data.append(parent_data)

# Create lists of unique values for other columns' filters
unique_problem = sorted(df["problem_name"].unique())
unique_opt = sorted(df["opt_global"].unique())
unique_fit_func = sorted(df["fit_func"].unique())
unique_noise = sorted(df["noise"].unique())
unique_type = sorted(df["algo_type"].unique())
# For "algo_names_list" filter, gather all distinct algo names from the DataFrame
unique_algos = sorted(df["algo_name"].unique())

# A small custom formatter that displays an array as a comma-separated string
# (this is a Tabulator "function" formatter in JS code)
formatter_code = """
function(cell, formatterParams, onRendered){
    let val = cell.getValue();
    if (Array.isArray(val)){
       return val.join(", ");
    }
    return val;
}
"""

columns = [
    {
        "title": "Problem",
        "field": "problem_name",
        "headerFilter": "select",
        "headerFilterParams": {"values": unique_problem, "multiple": True}
    },
    {
        "title": "Opt",
        "field": "opt_global",
        "headerFilter": "select",
        "headerFilterParams": {"values": unique_opt, "multiple": True}
    },
    {
        "title": "Function",
        "field": "fit_func",
        "headerFilter": "select",
        "headerFilterParams": {"values": unique_fit_func, "multiple": True}
    },
    {
        "title": "Noise",
        "field": "noise",
        "headerFilter": "select",
        "headerFilterParams": {"values": unique_noise, "multiple": True}
    },
    {
        "title": "Type",
        "field": "algo_type",
        "headerFilter": "select",
        "headerFilterParams": {"values": unique_type, "multiple": True}
    },
    {
        # The key part: "algo_names_list" is a LIST, which we filter with "in"
        "title": "Algorithm",
        "field": "algo_names_list",
        "formatter": {"type": "function", "code": formatter_code},
        "headerFilter": "select",
        # "in" filter checks if any selected value is in the row's array
        "headerFilterFunc": "in",
        "headerFilterParams": {
            "values": unique_algos,
            "multiple": True
        }
    },
    # Metrics
    {"title": "Unique Sols", "field": "n_unique_sols"},
    {"title": "Generations", "field": "n_gens"},
    {"title": "Evaluations", "field": "n_evals"},
    {"title": "Final Fit", "field": "final_fit"},
    {"title": "Max Fit", "field": "max_fit"},
    {"title": "Min Fit", "field": "min_fit"},
]

app = dash.Dash(__name__)

app.layout = html.Div([
    dt.DashTabulator(
        id="my-nested-table",
        data=nested_data,
        columns=columns,
        options={
            "dataTree": True,
            "dataTreeChildField": "children",
            "selectable": True,  # Enable row selection
        }
    ),
    html.Div(id="selected-tabulator-rows")
])

@app.callback(
    Output("selected-tabulator-rows", "children"),
    Input("my-nested-table", "selectedRows")
)
def display_selected_tabulator_rows(selected_rows):
    if not selected_rows:
        return "No rows selected."
    # selected_rows is a list of selected row objects:
    return f"Selected rows: {selected_rows}"

if __name__ == "__main__":
    app.run_server(debug=True)
