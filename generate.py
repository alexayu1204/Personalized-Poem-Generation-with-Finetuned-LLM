#!/usr/bin/env python3
import os
import re
import argparse
import subprocess
import tempfile
import shutil
import json
import sys

def sanitize_filename(s):
    """Return a safe filename version of the string."""
    return re.sub(r'\W+', '_', s)

def extract_epoch(checkpoint_path):
    """
    If checkpoint filename starts with an integer (e.g. "0001200_adapters.safetensors"),
    return that integer, otherwise return None.
    """
    base = os.path.basename(checkpoint_path)
    match = re.match(r'(\d+)_adapters\.safetensors$', base)
    if match:
        return int(match.group(1))
    return None

def get_checkpoint_files(adapter_dir, start_epoch):
    """
    Scan the adapter_dir for checkpoint files matching the pattern.
    Return a sorted list of tuples (epoch, full_path) for those meeting the minimum start_epoch.
    """
    adapter_files = []
    pattern = re.compile(r'(\d+)_adapters\.safetensors$')
    for fname in os.listdir(adapter_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch >= start_epoch:
                adapter_files.append((epoch, os.path.join(adapter_dir, fname)))
    return sorted(adapter_files, key=lambda x: x[0])

def build_prompt_variants(user_phase):
    """
    Build and return the list of 5 prompt strings based on the provided user phase.
    """
    templates = [
        # Variation 1: base prompt with no extra newline.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> "
        "<|im_start|>user {phase}<|im_end|> <|im_start|>assistant",
        # Variation 2: one newline after user phase.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> "
        "<|im_start|>user {phase}\\n<|im_end|> <|im_start|>assistant",
        # Variation 3: two newlines after user phase.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> "
        "<|im_start|>user {phase}\\n\\n<|im_end|> <|im_start|>assistant",
        # Variation 4: omit the closing tokens.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> "
        "<|im_start|>user {phase}",
        # Variation 5: two newlines, omit closing tokens.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> "
        "<|im_start|>user {phase}\\n\\n"
    ]
    return [template.format(phase=user_phase) for template in templates]

def run_generation(model, temp_adapter_dir, prompt):
    """
    Run the generation command with the given model, temporary adapter directory,
    and prompt. Return the generated text.
    """
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--adapter-path", temp_adapter_dir,
        "--prompt", prompt
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: Generation failed. Details: {e}\nOutput: {e.output}"

def update_output_files(unique_generations, txt_filename, json_filename):
    """
    Write current unique generation outputs to a TXT file and metadata to a JSON file.
    """
    with open(txt_filename, "w", encoding="utf-8") as txt_out:
        for gen_text in unique_generations.keys():
            txt_out.write(gen_text + "\n\n")
        txt_out.flush()
    meta_list = list(unique_generations.values())
    with open(json_filename, "w", encoding="utf-8") as json_out:
        json.dump(meta_list, json_out, ensure_ascii=False, indent=2)
        json_out.flush()

def main():
    parser = argparse.ArgumentParser(
        description="Generate text using multiple checkpoints and prompt variations, "
                    "deduplicate outputs, update files in real time, and print progress."
    )
    parser.add_argument("--model", required=True,
                        help="Path to the base model (e.g., mistralai/Mistral-7B-v0.1).")
    parser.add_argument("--adapter-dir", required=True,
                        help="Directory containing adapter checkpoint files and adapter_config.json.")
    parser.add_argument("--user-phase", required=True,
                        help="The user phase (e.g., '小雪'). This will be used in the prompt and output filenames.")
    parser.add_argument("--start-epoch", type=int, default=1000,
                        help="Minimum epoch number of checkpoints to use (default: 1000).")
    parser.add_argument("--checkpoints", nargs='+',
                        help="Manually specify a list of .safetensors checkpoint file paths to use (overrides scanning adapter-dir).")
    parser.add_argument("--config-path", default=None,
                        help="Path to adapter_config.json. Defaults to <adapter-dir>/adapter_config.json.")
    parser.add_argument("--output-txt", default=None,
                        help="Output TXT filename (default: <user_phase>.txt).")
    parser.add_argument("--output-json", default=None,
                        help="Output JSON filename (default: <user_phase>_metadata.json).")
    args = parser.parse_args()

    safe_phase = sanitize_filename(args.user_phase)
    if not args.output_txt:
        args.output_txt = f"{safe_phase}.txt"
    if not args.output_json:
        args.output_json = f"{safe_phase}_metadata.json"
    # Determine config file location.
    if args.config_path:
        config_path = args.config_path
    else:
        config_path = os.path.join(args.adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        print(f"Warning: adapter_config.json not found at {config_path}. Generation may fail.", file=sys.stderr)

    # Get list of checkpoints.
    checkpoint_list = []
    if args.checkpoints:
        # Use manually provided list.
        for cp in args.checkpoints:
            if os.path.isfile(cp):
                epoch = extract_epoch(cp)
                checkpoint_list.append((epoch, os.path.abspath(cp)))
            else:
                print(f"Checkpoint file {cp} does not exist; skipping.", file=sys.stderr)
        # If some checkpoints did not have a valid epoch, sort by filename.
        checkpoint_list.sort(key=lambda x: (x[0] if x[0] is not None else x[1]))
    else:
        # Scan adapter_dir for checkpoint files.
        checkpoint_list = get_checkpoint_files(args.adapter_dir, args.start_epoch)
    if not checkpoint_list:
        print("No checkpoint files found; exiting.", file=sys.stderr)
        return
    print(f"Found {len(checkpoint_list)} checkpoint(s).")

    prompt_variants = build_prompt_variants(args.user_phase)
    unique_generations = {}  # key: normalized generation text, value: metadata dict

    # Process each checkpoint.
    for epoch, cp_path in checkpoint_list:
        cp_display = f"{epoch}" if epoch is not None else os.path.basename(cp_path)
        print(f"Processing checkpoint {cp_display} from file: {cp_path}")
        for idx, prompt in enumerate(prompt_variants, start=1):
            print(f"  Using prompt variant {idx}")
            with tempfile.TemporaryDirectory() as temp_adapter_dir:
                # Copy checkpoint file as adapters.safetensors
                temp_cp_path = os.path.join(temp_adapter_dir, "adapters.safetensors")
                shutil.copy(cp_path, temp_cp_path)
                # Also copy adapter_config.json if it exists.
                if os.path.isfile(config_path):
                    shutil.copy(config_path, os.path.join(temp_adapter_dir, "adapter_config.json"))
                else:
                    print("    Warning: adapter_config.json not found; skipping config copy.", file=sys.stderr)
                # Run generation.
                generation = run_generation(args.model, temp_adapter_dir, prompt)
                norm_generation = generation.strip()
                if not norm_generation:
                    print("    Skipping empty generation output.")
                    continue
                # Check for duplicates.
                if norm_generation in unique_generations:
                    prev_meta = unique_generations[norm_generation]
                    prev_cp = prev_meta.get("checkpoint_number") or prev_meta.get("checkpoint_path")
                    print(f"    Duplicate generation: checkpoint {cp_display} is generating same as {prev_cp}.")
                    continue
                # Build metadata for this generation.
                meta = {
                    "prompt_phase": args.user_phase,
                    "prompt": prompt,
                    "checkpoint_number": epoch if epoch is not None else cp_display,
                    "checkpoint_path": cp_path,
                    "generation": norm_generation,
                    "prompt_variant_index": idx
                }
                unique_generations[norm_generation] = meta
                # Print the unique generation to terminal.
                print(f"    [Unique] Checkpoint {cp_display}, prompt variant {idx}:")
                print(norm_generation)
                print("-" * 40)
                # Update output files in real time.
                update_output_files(unique_generations, args.output_txt, args.output_json)
    print(f"All generations processed. Final outputs saved to: {args.output_txt} and {args.output_json}")

if __name__ == "__main__":
    main()
