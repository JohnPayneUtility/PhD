import dash
from dash import html, dcc, Output, Input
import dash_tabulator as dt
import pandas as pd

# For demonstration, we'll create some sample data.
data = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
    "Score": [85, 92, 78, 88, 95, 80],
    "Group": ["A", "B", "A", "B", "A", "B"]
}).to_dict("records")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Dash Tabulator (with Row Selection Captured via Clientside Callback)"),
    dt.DashTabulator(
        id="tabulator",
        data=data,
        columns=[{"title": col, "field": col, "headerFilter": "input"} for col in ["Name", "Score", "Group"]],
        options={
            "layout": "fitData",
            "pagination": "local",
            "paginationSize": 10,
            "selectable": True  # enable row selection in the UI
        },
    ),
    # A hidden store to capture selected rows from Tabulator
    dcc.Store(id="selected-store"),
    # An interval component to trigger the clientside callback periodically
    dcc.Interval(id="interval", interval=1000, n_intervals=0),
    html.Div(id="output-selected")
])

# Clientside callback (written in JavaScript) to update the store with the selected rows.
app.clientside_callback(
    """
    function(n_intervals) {
        var tabElem = document.getElementById("tabulator");
        if (tabElem) {
            // Log the element to help debug:
            console.log("Tabulator element:", tabElem);
            if (tabElem.__dashTabulatorInstance__) {
                console.log("Tabulator instance found:", tabElem.__dashTabulatorInstance__);
                var selectedData = tabElem.__dashTabulatorInstance__.getSelectedData();
                console.log("Selected data:", selectedData);
                return selectedData;
            } else {
                console.log("No Tabulator instance attached to the element.");
            }
        }
        return [];
    }
    """,
    Output("selected-store", "data"),
    Input("interval", "n_intervals")
)

# A Python callback to display the selected rows stored in the dcc.Store.
@app.callback(
    Output("output-selected", "children"),
    Input("selected-store", "data")
)
def display_selected(selected):
    if not selected:
        return "No rows selected."
    return f"Selected Rows: {selected}"

if __name__ == '__main__':
    app.run_server(debug=True)
