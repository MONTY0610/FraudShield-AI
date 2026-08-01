import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Preprocess the credit card dataset.
    """

    # Check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale Time and Amount
    scaler = StandardScaler()

    X["Time"] = scaler.fit_transform(X[["Time"]])
    X["Amount"] = scaler.fit_transform(X[["Amount"]])

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_data("data/creditcard.csv/creditcard.csv")

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("\nDataset Loaded Successfully!")

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")