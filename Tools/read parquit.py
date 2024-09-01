import pandas as pd
import os

def save_source_code_to_solidity(directory):
    """
    This function reads source code from all parquet files in a given directory
    and saves each snippet to a separate Solidity file.

    Args:
        directory (str): Path to the directory containing parquet files.
    """

    try:
        # List all files in the directory
        files = os.listdir(directory)
        # Filter out parquet files
        parquet_files = [f for f in files if f.endswith('.parquet')]

        # Check if any parquet files found
        if not parquet_files:
            print("No Parquet files found in the directory.")
            return

        # Initialize a counter for file naming
        file_counter = 1

        for parquet_file in parquet_files:
            full_path = os.path.join(directory, parquet_file)
            df = pd.read_parquet(full_path)
            
            # Print the columns of the DataFrame
            print(f"Columns in {parquet_file}: {df.columns.tolist()}")

            if 'source_codes' not in df.columns:
                print(f"'source_code' column not found in {parquet_file}. Skipping this file.")
                continue
            
            for i, row in df.iterrows():
                source_code = row['source_codes']
                filename = f"source_code_{file_counter}.sol"  # Use file_counter for unique filenames
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(source_code)
                file_counter += 1  # Increment counter for the next file
            print(f"Source code snippets from {parquet_file} saved to separate Solidity files.")

    except FileNotFoundError as e:
        print(f"Error: Directory not found. Please check the path: {directory}")

# Example usage
directory = "C:\\Users\\Cody\\Desktop\\FYP Project\\Code\\AI Model\\Model Training"
save_source_code_to_solidity(directory)
