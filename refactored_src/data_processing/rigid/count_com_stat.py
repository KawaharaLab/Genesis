import pandas as pd
import os

# Define the constants
DATA_DIR = "/home/user/Genesis/data"
DATA_TYPE = "eval"
com = "_com"
N = 15

# Construct the file path
file_path = os.path.join(DATA_DIR, DATA_TYPE, f"{DATA_TYPE}{com}.csv")
file_path = os.path.join(DATA_DIR, DATA_TYPE, f"{DATA_TYPE}_thin_{N}pct{com}.csv")

try:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the "annotation" column exists
    if 'annotation' not in df.columns:
        print(f"Error: The 'annotation' column was not found in the CSV file.")
    else:
        # Count the occurrences of "far" and "near" using string containment
        # The `.str.contains()` method checks if the string contains the specified substring.
        # `na=False` treats any missing values as False to avoid errors.
        far_count = df['annotation'].str.contains('far', na=False).sum()
        near_count = df['annotation'].str.contains('near', na=False).sum()

        # Print the results
        print(f"Number of 'far' annotations: {far_count}")
        print(f"Number of 'near' annotations: {near_count}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")