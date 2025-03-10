import pandas as pd

data = pd.read_csv('dataset.csv')
print("Toutes les colonnes du CSV :")
print(data.columns.tolist())
