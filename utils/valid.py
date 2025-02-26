import json
import sys
import os

def validate_json(file_path, required_keys, add_missing=False, default_values=None):
    """
    Validates the JSON format and verifies the presence of required keys in each object.

    Args:
        file_path (str): Path to the JSON file.
        required_keys (list): List of keys that must be present in each JSON object.
        add_missing (bool): If True, adds missing keys with default values.
        default_values (dict): Dictionary of default values for missing keys.

    Returns:
        bool: True if JSON is valid and all required keys are present, False otherwise.
    """
    if default_values is None:
        default_values = {}

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    # Load JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return False
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return False

    # Check if data is a list
    if not isinstance(data, list):
        print("Error: JSON data is not an array of objects.")
        return False

    all_valid = True
    missing_keys_info = []

    # Iterate through each object and check for required keys
    for index, obj in enumerate(data, start=1):
        if not isinstance(obj, dict):
            print(f"Error: Item at index {index} is not a JSON object.")
            all_valid = False
            continue

        missing_keys = [key for key in required_keys if key not in obj]
        if missing_keys:
            all_valid = False
            missing_keys_info.append((index, missing_keys))
            if add_missing:
                for key in missing_keys:
                    obj[key] = default_values.get(key, None)

    # Report missing keys
    if missing_keys_info:
        print("Missing Keys Detected:")
        for item in missing_keys_info:
            print(f" - Object at index {item[0]} is missing keys: {', '.join(item[1])}")

        if add_missing:
            # Backup the original file
            backup_path = file_path + ".backup"
            try:
                if not os.path.exists(backup_path):
                    os.rename(file_path, backup_path)
                    print(f"Backup created at '{backup_path}'.")
                else:
                    print(f"Backup already exists at '{backup_path}'.")
            except Exception as e:
                print(f"Failed to create backup: {e}")
                return False

            # Save the updated data back to the original file
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Missing keys added with default values in '{file_path}'.")
            except Exception as e:
                print(f"Failed to write updated JSON data: {e}")
                # Attempt to restore from backup
                if os.path.exists(backup_path):
                    os.rename(backup_path, file_path)
                    print("Restored the original file from backup.")
                return False

    if all_valid:
        print("JSON is valid. All required keys are present in each object.")
    else:
        if not add_missing:
            print("JSON validation failed due to missing keys.")
        else:
            print("JSON validation completed. Missing keys were added.")

    return all_valid

def main():
    # Path to your JSON file
    json_file_path = 'overall_poems.json'  # Modify if your file is located elsewhere

    # Define the required keys
    required_keys = ["adaptor", "outer_idx", "inner_idx", "image", "response",
                     "accepted", "edited", "deleted", "id"]

    # Define default values for missing keys if you choose to add them
    default_values = {
        "deleted": False,
        "id": None  # Or any default value you'd like for 'id'
    }

    # Parse command-line arguments for flexibility
    import argparse

    parser = argparse.ArgumentParser(description="Validate JSON format and verify required keys.")
    parser.add_argument('file', nargs='?', default=json_file_path,
                        help='Path to the JSON file to validate.')
    parser.add_argument('--add-missing', action='store_true',
                        help='Add missing keys with default values.')
    parser.add_argument('--default-deleted', type=str, default='false',
                        help='Default value for "deleted" key (true/false).')
    parser.add_argument('--default-id', type=int, default=None,
                        help='Default value for "id" key.')

    args = parser.parse_args()

    # Update default_values based on arguments
    if args.default_deleted.lower() == 'true':
        default_values["deleted"] = True
    elif args.default_deleted.lower() == 'false':
        default_values["deleted"] = False
    else:
        print("Warning: Invalid value for --default-deleted. Using False as default.")
        default_values["deleted"] = False

    default_values["id"] = args.default_id

    # Run validation
    validate_json(args.file, required_keys, add_missing=args.add_missing,
                 default_values=default_values)

if __name__ == "__main__":
    main()

