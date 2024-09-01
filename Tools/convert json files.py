import os

def convert_json_files(folder_path):
    """
    Converts JSON files in the specified folder, making the following changes:
    - Replaces single quotes with double quotes
    - Converts 'False' to 'false'
    - Converts 'True' to 'true'
    - Converts 'None' to 'null'

    Args:
        folder_path: The path to the folder containing the JSON files.
    """

    for filename in os.listdir(folder_path):
        try:
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'r') as f:
                    data = f.read()

                # Perform the necessary transformations
                data = data.replace("'", '"').replace("False", "false").replace("True", "true").replace("None", "null")

                # Write the modified data back to the file
                with open(file_path, 'w') as f:
                    f.write(data)

                print(f"Converted file: {filename}")
        except:
            print(f"Error converting file: {filename}")
            os.remove(file_path)

if __name__ == "__main__":
    folder_path = "./AI Model/Model Training/AST/Verified"  # Replace with the actual path to your folder
    convert_json_files(folder_path)
