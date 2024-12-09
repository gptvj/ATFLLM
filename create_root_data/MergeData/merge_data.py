import json

# Paths to the two JSON files
file1_path = '../Law2Contract/Data/LawContractAPI_full.json'
file2_path = '../Contract2Law/Data/find_law_gemini_contract_formatted.json'
output_path = 'merge_data_full.json'

# Function to read JSON content from a file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to write JSON content to a file
def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Function to merge two JSON objects (lists of dictionaries in this case)
def merge_json(file1, file2):
    merged_data = file1 + file2  # Combine the two lists
    return merged_data

# Read data from the two JSON files
data1 = read_json(file1_path)
data2 = read_json(file2_path)

# Merge the data from the two files
merged_data = merge_json(data1, data2)

# Write the merged data to a new file
write_json(output_path, merged_data)

print(f"Merged data has been saved to {output_path}")
