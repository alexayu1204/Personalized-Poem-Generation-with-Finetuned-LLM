#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
import tempfile
import shutil

def sanitize_filename(s):
    # Replace non-alphanumeric characters with underscores.
    return re.sub(r'\W+', '_', s)

def main():
    parser = argparse.ArgumentParser(
        description="Generate outputs using each intermediate adapter checkpoint by temporarily renaming it to adapters.safetensors."
    )
    parser.add_argument("--model", required=True,
                        help="Path to the base model (e.g., mistralai/Mistral-7B-v0.1).")
    parser.add_argument("--adapter-dir", required=True,
                        help="Directory containing adapter checkpoint files.")
    parser.add_argument("--prompt", required=True,
                        help="The prompt to use for generation.")
    parser.add_argument("--start-epoch", type=int, default=1000,
                        help="Minimum epoch number of checkpoints to use (default: 1000).")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: sanitized prompt + '.txt').")
    args = parser.parse_args()

    # Set the output file name based on the prompt if not provided.
    if args.output is None:
        args.output = f"{sanitize_filename(args.prompt)}.txt"

    # Gather adapter checkpoint files with pattern like "0001200_adapters.safetensors"
    adapter_files = []
    pattern = re.compile(r'(\d+)_adapters\.safetensors$')
    for fname in os.listdir(args.adapter_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch >= args.start_epoch:
                adapter_files.append((epoch, os.path.join(args.adapter_dir, fname)))

    if not adapter_files:
        print(f"No adapter checkpoint files found with epoch >= {args.start_epoch} in {args.adapter_dir}")
        return

    # Sort checkpoint files by epoch (ascending)
    adapter_files.sort(key=lambda x: x[0])
    print(f"Found {len(adapter_files)} checkpoints from epoch {adapter_files[0][0]} to {adapter_files[-1][0]}.")

    # Open the output file for concatenated generations.
    with open(args.output, "w") as outf:
        for epoch, adapter_file in adapter_files:
            print(f"Processing epoch {epoch} using adapter file: {adapter_file}")
            # Create a temporary directory where we copy the adapter file as 'adapters.safetensors'
            with tempfile.TemporaryDirectory() as temp_adapter_dir:
                temp_adapter_path = os.path.join(temp_adapter_dir, "adapters.safetensors")
                shutil.copy(adapter_file, temp_adapter_path)
                print(f"Copied {adapter_file} to temporary directory as {temp_adapter_path}")

                # Build the generation command, pointing --adapter-path to the temporary directory.
                cmd = [
                    "mlx_lm.generate",
                    "--model", args.model,
                    "--adapter-path", temp_adapter_dir,
                    "--prompt", args.prompt
                ]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    generation = result.stdout.strip()
                except subprocess.CalledProcessError as e:
                    generation = (f"Error generating output for epoch {epoch}.\n"
                                  f"Error details: {e}\nOutput: {e.output}\n")
                # Write a header and the generation output to the file.
                outf.write(f"==== Epoch {epoch} (Adapter file: {adapter_file}) ====\n")
                outf.write(generation + "\n\n")
    print(f"All generations have been concatenated and saved to {args.output}")

if __name__ == "__main__":
    main()
