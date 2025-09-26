

import pandas as pd
import glob
import os

# Define the directory where your CSV files are located.
# Change this to the actual path on your system.
# DATA_DIR = "/home/user/Genesis/data/eval_heavy/eval_new.csv"
DATA_DIR = "/home/user/Genesis/data/train_old/train_new.csv"

def analyze_source_types(file_path: str):
    """
    Analyzes the percentage of 'source_type' 0 and 1 in each CSV file.
    """
    # csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # if not csv_files:
    #     print(f"No CSV files found in '{directory}'.")
    #     return

    
    try:
        df = pd.read_csv(file_path)

        if 'source_type' not in df.columns or df.empty:
            print(f"\nSkipping '{os.path.basename(file_path)}': 'source_type' column missing or file is empty.")
            return

        # Use value_counts() to get counts and normalize for percentages
        percentages = df['source_type'].value_counts(normalize=True) * 100

        print(f"\n--- Analysis for '{os.path.basename(file_path)}' ---")
        print(f"Total rows: {len(df)}")
        print(f"Percentage of 'train_old' (0): {percentages.get(0, 0):.2f}%")
        print(f"Percentage of 'eval_heavy' (1): {percentages.get(1, 0):.2f}%")
        
    except Exception as e:
        print(f"\nError processing '{os.path.basename(file_path)}': {e}")

def analyze_labels(file_path: str):
    """
    Analyzes the count and percentage of each label in a CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        if 'label' not in df.columns or df.empty:
            print(f"Skipping '{os.path.basename(file_path)}': 'label' column missing or file is empty.")
            return

        # Get the count of each unique label
        label_counts = df['label'].value_counts()
        
        # Get the percentage of each unique label
        label_percentages = df['label'].value_counts(normalize=True) * 100

        print(f"\n--- Analysis for '{os.path.basename(file_path)}' ---")
        print(f"Total rows: {len(df)}")
        print("\nLabel Counts:")
        print(label_counts.to_string()) # Use .to_string() for clean output
        print("\nLabel Percentages:")
        print(label_percentages.to_string(float_format="%.2f%%"))

    except Exception as e:
        print(f"\nError processing '{os.path.basename(file_path)}': {e}")

# Run the analysis
# analyze_source_types(DATA_DIR)
analyze_labels(DATA_DIR)