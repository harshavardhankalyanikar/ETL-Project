import os
import pandas as pd
from extract_titanic import extract_titani


def transform_data(raw_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_dir = os.path.join(base_dir, "data", "staged")
    os.makedirs(staged_dir, exist_ok=True)
    df = pd.read_csv(raw_path)

    # 1.Handling  missing values
    numeric_cols = ["age", "fare"]

    # filling the missing values with the median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "embarked" in df.columns:
        df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # 2.Feature Engineering
    if "age" in df.columns and "fare" in df.columns:
        df["age_fare_ratio"] = df["age"] / (df["fare"] + 1)  # +1 to avoid division by zero
    
    if "age" in df.columns:
        df["is_adult"] = (df["age"] >= 18).astype(int)
    
    if "survived" in df.columns:
        df["survived"] = df["survived"].astype(int)

    # 3. Drop unnecessary columns
    df.drop(columns=[], inplace=True, errors="ignore")

    # 4.save  data
    staged_path = os.path.join(staged_dir, "titanic_transformed.csv")
    df.to_csv(staged_path, index=False)

    print(f"Data transformed and saved to {staged_path}")
    return staged_path


if __name__ == "__main__":
    from extract_titanic import extract_titanic
    raw_path = extract_titanic()
    transform_data(raw_path)