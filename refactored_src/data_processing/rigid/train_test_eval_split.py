import pandas as pd

def find_split_point(df: pd.DataFrame, split_ratio: float = 0.9) -> int:
    """
    Finds the index for a train/test split at a given ratio,
    ensuring that the split does not occur in the middle of a 'csv_path' group.

    Args:
        df (pd.DataFrame): The input DataFrame, sorted by a relevant column.
        split_ratio (float): The desired ratio for the training set (e.g., 0.9 for 90%).

    Returns:
        int: The index of the first row of the test set.
    """
    if 'csv_path' not in df.columns:
        raise ValueError("DataFrame must contain a 'csv_path' column.")

    # Calculate the target index for the split
    target_index = int(len(df) * split_ratio)

    # Find the next index where 'csv_path' changes
    current_path = df.at[target_index, 'csv_path']
    for i in range(target_index, len(df)):
        if df.at[i, 'csv_path'] != current_path:
            return i
            
    # If the target is at the end of the last object, return the end of the DataFrame
    return len(df)

# Example Usage:
# Assuming 'out' is your combined DataFrame
out = pd.read_csv("Genesis/data/train_eval_mixed.csv")

# Ensure the DataFrame is sorted by csv_path to group objects
# out = out.sort_values(by='csv_path')

# Find the split index
split_index = find_split_point(out)

# Split the data
train_df = out.iloc[:split_index] # first 90% of objects
eval_df = out.iloc[split_index:]

# Optional: Print split information
print(f"Train set size: {len(train_df)} rows")
print(f"Evaluation set size: {len(eval_df)} rows")
print(f"Split ratio: {len(train_df) / len(out):.2f}")

# Save the splits to CSV files
train_df.to_csv("/home/user/Genesis/data/train_old/train_new.csv", index=False)
eval_df.to_csv("/home/user/Genesis/data/eval_heavy/eval_new.csv", index=False)