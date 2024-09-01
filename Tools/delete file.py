import os

# Base file path
base_path = 'C:/Users/Cody/Desktop/FYP Project/Code/AI Model/Model Training/Contracts for training/Re-entrancy/'

# Path to the text file containing file paths
file_paths_txt = 'C://Users//Cody//Desktop//FYP Project//Code//AI Model//Model Training//error_files.txt'

# Read the file paths from the text file
with open(file_paths_txt, 'r') as file:
    file_paths = file.readlines()

# Remove leading/trailing whitespaces and newlines from each file path
file_paths = [path.strip() for path in file_paths]

# Combine each file path with the base path
file_paths = [os.path.join(base_path, path) for path in file_paths]

# Delete each file
for path in file_paths:
    try:
        os.remove(path)
        print(f"Deleted file: {path}")
    except OSError as e:
        print(f"Error deleting file: {path} - {e}")