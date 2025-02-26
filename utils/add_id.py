import json
import sys
import os

def update_json(file_path):
    # Backup the original file
    backup_path = file_path + ".backup"
    try:
        if not os.path.exists(backup_path):
            os.rename(file_path, backup_path)
            print(f"Backup created at {backup_path}")
        else:
            print(f"Backup already exists at {backup_path}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        sys.exit(1)

    # Load the JSON data from the backup
    try:
        with open(backup_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Failed to read JSON data: {e}")
        sys.exit(1)

    # Check if the data is a list
    if not isinstance(data, list):
        print("JSON data is not a list of objects.")
        sys.exit(1)

    # Iterate through each object and add 'deleted' and 'id' keys
    for index, item in enumerate(data, start=1):
        # Add the 'deleted' key if it doesn't exist
        if 'deleted' not in item:
            item['deleted'] = False  # Set default value as needed

        # Add or update the 'id' key
        item['id'] = index

    # Save the updated JSON data back to the original file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Successfully updated {file_path} with 'deleted' and 'id' keys.")
    except Exception as e:
        print(f"Failed to write updated JSON data: {e}")
        # Attempt to restore from backup
        os.rename(backup_path, file_path)
        print("Restored the original file from backup.")
        sys.exit(1)

if __name__ == "__main__":
    json_file_path = 'overall_poems.json'  # Modify if your file is located elsewhere
    update_json(json_file_path)

