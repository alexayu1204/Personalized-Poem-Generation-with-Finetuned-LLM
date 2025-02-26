#!/usr/bin/env python3
import json
import argparse
import sys

def flatten_json_structure(data):
    """
    Given a dictionary where each value is a list,
    flatten all lists into a single list.
    """
    flattened_list = []
    if not isinstance(data, dict):
        print("Error: Input JSON must be a dictionary at the top level.")
        sys.exit(1)
    for key, value in data.items():
        if isinstance(value, list):
            flattened_list.extend(value)
        else:
            print(f"Warning: The value for key '{key}' is not a list and will be skipped.")
    return flattened_list

def main():
    parser = argparse.ArgumentParser(
        description="Flatten a JSON file (with dict values as lists) into a single list of dictionaries."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "-o", "--output",
        default="structured.json",
        help="Path to the output JSON file (default: structured.json)."
    )
    args = parser.parse_args()

    # Load the input JSON file
    try:
        with open(args.input, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except Exception as e:
        print(f"Error loading JSON file '{args.input}': {e}")
        sys.exit(1)

    # Flatten the JSON structure
    flattened = flatten_json_structure(data)

    # Write the flattened list to the output file
    try:
        with open('structured.json', 'w', encoding='utf-8') as outfile:
            json.dump(flattened, outfile, ensure_ascii=False, indent=4)
        print(f"Successfully written flattened JSON to '{args.output}'.")
    except Exception as e:
        print(f"Error writing JSON file '{args.output}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()