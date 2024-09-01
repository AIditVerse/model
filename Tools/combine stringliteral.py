import re
import os
import glob

def combine_string_literals(text):
    # Regular expression to find all "stringLiteral" occurrences
    pattern = re.compile(r'"type":\s*"stringLiteral",\s*"value":\s*"([^"]*)"\s*(("([^"]*)"\s*)*)')

    def combine_values(match):
        # Extract the initial value
        combined_value = match.group(1)
        
        # Extract additional values if any
        additional_values = re.findall(r'"([^"]*)"', match.group(2))
        
        # Combine all the values into one string
        combined_value += ' ' + ' '.join(additional_values)
        
        # Clean up the combined string (remove extra spaces and newlines)
        combined_value = combined_value.replace('\n', ' ').strip()
        
        # Return the combined result formatted as the JSON snippet
        return f'"type": "stringLiteral", "value": "{combined_value}"'

    # Substitute the found pattern with the combined value string
    cleaned_text = pattern.sub(combine_values, text)
    return cleaned_text

def process_file(input_file, output_file):
    try:
        # Read the file content as text
        with open(input_file, 'r') as f:
            text = f.read()
        
        # Combine string literals
        cleaned_text = combine_string_literals(text)
        
        # Write the cleaned text back to a new file
        with open(output_file, 'w') as f:
            f.write(cleaned_text)
        
        print(f"Processed and saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Find all JSON files in the input folder
    json_files = glob.glob(os.path.join(input_folder, '*.json'))

    for json_file in json_files:
        # Define the output file path
        output_file = os.path.join(output_folder, os.path.basename(json_file))
        
        # Process each JSON file
        process_file(json_file, output_file)

# Example usage
input_folder = './Model Training/AST/Verified/'  # Replace with your input folder path
output_folder = './Model Training/AST/Verified-modified/'  # Replace with your output folder path

process_folder(input_folder, output_folder)
