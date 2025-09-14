import pandas as pd
import os

# Define the constants
DATA_DIR = "/home/user/Genesis/data"
DATA_TYPE = "train"
weight = "_weight"
N = 20

# Construct the file path
file_path = os.path.join(DATA_DIR, DATA_TYPE, f"{DATA_TYPE}{weight}.csv")
file_path = os.path.join(DATA_DIR, DATA_TYPE, f"{DATA_TYPE}_thin_{N}pct{weight}.csv")

try:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the "annotation" column exists
    if 'annotation' not in df.columns:
        print(f"Error: The 'annotation' column was not found in the CSV file.")
    else:
        # Count the occurrences of "light" and "heavy" using string containment
        # The `.str.contains()` method checks if the string contains the specified substring.
        # `na=False` treats any missing values as False to avoid errors.
        light_count = df['annotation'].str.contains('light', na=False).sum()
        heavy_count = df['annotation'].str.contains('heavy', na=False).sum()

        # Print the results
        print(f"Number of 'light' annotations: {light_count}")
        print(f"Number of 'heavy' annotations: {heavy_count}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")