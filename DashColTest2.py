import dash
from dash import dash_table, html, Input, Output
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Score": [85, 92, 78, 88]
})

app = dash.Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(
        id="data-table",
        data=df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        row_selectable="multi",   # Enable multiple row selection
        selected_rows=[],         # Start with no rows selected
        page_size=10,             # Display up to 10 rows per page
        style_table={"overflowX": "auto"},
    ),
    html.Div(id="selected-rows-output")
])

@app.callback(
    Output("selected-rows-output", "children"),
    Input("data-table", "selected_rows"),
    Input("data-table", "data")
)
def display_selected_rows(selected_rows, rows):
    if not selected_rows:
        return "No rows selected."
    # Retrieve the data for each selected row using the indices
    selected_data = [rows[i] for i in selected_rows]
    return f"Selected rows: {selected_data}"

if __name__ == '__main__':
    app.run_server(debug=True)
