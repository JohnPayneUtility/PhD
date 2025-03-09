import pandas as pd

df = pd.read_pickle('results_LON.pkl')

# Display the DataFrame
print(df.columns)
print(df)
