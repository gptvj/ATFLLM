import json

# Paths for input and output files
input_file_path = 'merge_data_full.json'
output_file_path = 'vj_database.json'

# Function to reformat the merged JSON file
def reformat_json(input_file, output_file):
    # Read the merged JSON file
    with open(input_file, 'r', encoding='utf-8') as file:
        merged_data = json.load(file)
    
    # Create a new structure for the reformatted JSON
    formatted_data = {
        "_name_": "test",  # Set the name field as required
        "_date_time_": "2024-07-17 09:57:21",  # Set the date_time field (you can use the current date-time dynamically)
        "_count_": len(merged_data),  # Count the number of items
        "items": merged_data  # Add all items directly
    }
    
    # Write the formatted data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(formatted_data, file, ensure_ascii=False, indent=4)
    
    print(f"Formatted data has been saved to {output_file}")

# Reformat the JSON
reformat_json(input_file_path, output_file_path)
