import pandas as pd

from src.predict import predict

# Read only first 5 transactions
df = pd.read_csv("data/creditcard.csv/creditcard.csv").head()

# Remove target column
df = df.drop("Class", axis=1)

results = predict(df)

print(results)