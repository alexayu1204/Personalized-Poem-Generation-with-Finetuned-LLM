import json

# Path to your JSON file
json_file_path = 'overall_poems.json'

# Load the JSON data
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Check if the data is a list
if isinstance(data, list):
    # Iterate through each object in the list
    for item in data:
        # Add the 'deleted' key if it doesn't exist
        if 'deleted' not in item:
            item['deleted'] = False  # Set default value as needed
else:
    print("JSON data is not a list of objects.")
    exit(1)

# Save the updated JSON data back to the file
with open(json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print(f"Successfully added 'deleted' key to each object in {json_file_path}.")

