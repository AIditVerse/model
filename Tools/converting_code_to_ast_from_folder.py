import os
import pprint
from solidity_parser import parser

# Function to parse Solidity files and save ASTs
def parse_and_save_ast(folder_path, ast_folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter Solidity files
    solidity_files = [f for f in files if f.endswith('.sol')]
    
    # Iterate over each Solidity file
    for file_name in solidity_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # Parse the file and get the AST
            source_unit = parser.parse_file(file_path, loc=True)
        except Exception as e:
            print(f"Error parsing {file_name}: {e}")
            continue  # Skip to the next file
        
        # Create a directory to store AST files if it doesn't exist
        ast_folder = os.path.join(ast_folder_path, 'UR')
        os.makedirs(ast_folder, exist_ok=True)
        
        # Save the AST to a file
        output_file_path = os.path.join(ast_folder, file_name.split('.')[0] + '_ast.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            pprint.pprint(source_unit, stream=f)

        print(f"Processed {file_name}")

# Provide the folder path here
folder_path = './Code/Contracts for training/Re-entrancy'  # Adjust the path as per your folder location
ast_folder_path = './AST'

# Call the function to parse and save ASTs
parse_and_save_ast(folder_path, ast_folder_path)
