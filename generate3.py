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
    If the checkpoint filename starts with an integer (e.g. "0001200_adapters.safetensors"),
    return that integer; otherwise, return None.
    """
    base = os.path.basename(checkpoint_path)
    match = re.match(r'(\d+)_adapters\.safetensors$', base)
    if match:
        return int(match.group(1))
    return None

def get_checkpoint_files(adapter_dir, start_epoch):
    """
    Scan adapter_dir for checkpoint files matching the pattern.
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

def build_prompt_variants(phrase):
    """
    Build and return a list of 5 prompt strings based on the provided prompt phrase.
    The variants include slight differences (adding one or two newlines or omitting closing tokens).
    """
    templates = [
        # Variation 1: base prompt with no extra newline.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}<|im_end|> <|im_start|>assistant",
        # # Variation 2: one newline after the phrase.
        # "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        # "<|im_start|>user {phrase}\\n<|im_end|> <|im_start|>assistant",
        # # # Variation 3: two newlines after the phrase.
        # # "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        # # "<|im_start|>user {phrase}\\n\\n<|im_end|> <|im_start|>assistant",
        # # # Variation 4: omit the closing tokens.
        # # "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        # # "<|im_start|>user {phrase}",
        # # Variation 5: two newlines and omit the closing tokens.
        # "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        # "<|im_start|>user {phrase}\\n\\n"
    ]
    return [template.format(phrase=phrase) for template in templates]

def run_generation(model, temp_adapter_dir, prompt):
    """
    Run the mlx_lm.generate command with the given model, temporary adapter directory,
    and prompt. Return the raw generated output.
    """
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--adapter-path", temp_adapter_dir,
        "--prompt", prompt,
        "--max-tokens", "256"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: Generation failed. Details: {e}\nOutput: {e.output}"

def parse_generation_output(output):
    """
    Parse the raw output from mlx_lm.generate.
    Expected format:
      ==========
      <actual generated text>
      ==========
      Prompt: <N> tokens, ... 
      Generation: <M> tokens, ...
      Peak memory: ...
      
    Returns a tuple: (body, prompt_token, generate_token)
    The body is cleaned by removing all whitespace (spaces, tabs, etc.) except newlines.
    If parsing fails, returns (output.strip(), None, None).
    """
    parts = output.split("==========")
    if len(parts) >= 3:
        body = parts[1].strip()
        # Remove all whitespace (except newline characters)
        body = re.sub(r"[^\S\n]+", "", body)
        extra = parts[2]
        prompt_token = None
        generate_token = None
        prompt_match = re.search(r"Prompt:\s*(\d+)\s*tokens", extra)
        if prompt_match:
            prompt_token = int(prompt_match.group(1))
        gen_match = re.search(r"Generation:\s*(\d+)\s*tokens", extra)
        if gen_match:
            generate_token = int(gen_match.group(1))
        return body, prompt_token, generate_token
    return output.strip(), None, None

def update_output_files(unique_generations, txt_filename, json_filename):
    """
    Write current unique generation bodies to a TXT file and metadata to a JSON file.
    """
    with open(txt_filename, "w", encoding="utf-8") as txt_out:
        for body in unique_generations.keys():
            txt_out.write(body + "\n\n")
        txt_out.flush()
    meta_list = list(unique_generations.values())
    with open(json_filename, "w", encoding="utf-8") as json_out:
        json.dump(meta_list, json_out, ensure_ascii=False, indent=2)
        json_out.flush()

def process_prompt(phrase, checkpoint_list, model, config_path, txt_filename, json_filename, global_dict=None):
    """
    Process a single prompt phrase:
      - Build prompt variants from the phrase.
      - For each checkpoint and variant, run generation.
      - Deduplicate outputs (using a global dictionary if provided).
      - Print unique generations (or duplicate notices) and update output files.
    Returns the dictionary (global or local) of unique generations.
    """
    unique_generations = global_dict if global_dict is not None else {}
    prompt_variants = build_prompt_variants(phrase)
    
    for epoch, cp_path in checkpoint_list:
        cp_display = f"{epoch}" if epoch is not None else os.path.basename(cp_path)
        print(f"Processing checkpoint {cp_display} from file: {cp_path} for prompt: {phrase}")
        for idx, prompt in enumerate(prompt_variants, start=1):
            print(f"  Using prompt variant {idx}")
            with tempfile.TemporaryDirectory() as temp_adapter_dir:
                # Copy checkpoint file as adapters.safetensors.
                temp_cp_path = os.path.join(temp_adapter_dir, "adapters.safetensors")
                shutil.copy(cp_path, temp_cp_path)
                # Copy adapter_config.json if available.
                if os.path.isfile(config_path):
                    shutil.copy(config_path, os.path.join(temp_adapter_dir, "adapter_config.json"))
                else:
                    print("    Warning: adapter_config.json not found; skipping config copy.", file=sys.stderr)
                raw_output = run_generation(model, temp_adapter_dir, prompt)
                body, prompt_token, generate_token = parse_generation_output(raw_output)
                norm_body = body.strip()
                if not norm_body:
                    print("    Skipping empty generation output.")
                    continue
                if norm_body in unique_generations:
                    prev_meta = unique_generations[norm_body]
                    prev_cp = prev_meta.get("checkpoint_number") or prev_meta.get("checkpoint_path")
                    print(f"    Duplicate generation: checkpoint {cp_display} is generating same as {prev_cp}.")
                    continue
                meta = {
                    "title": phrase,
                    "prompt": prompt,
                    "checkpoint_number": epoch if epoch is not None else cp_display,
                    "checkpoint_path": cp_path,
                    "body": norm_body,
                    "prompt_token": prompt_token,
                    "generate_token": generate_token,
                    "prompt_variant_index": idx
                }
                unique_generations[norm_body] = meta
                print(f"    [Unique] Checkpoint {cp_display}, prompt variant {idx}:")
                print("==========")
                print(norm_body)
                print("==========")
                print(f"Prompt tokens: {prompt_token}, Generation tokens: {generate_token}")
                print("-" * 40)
                update_output_files(unique_generations, txt_filename, json_filename)
    return unique_generations

def main():
    parser = argparse.ArgumentParser(
        description="Generate text using multiple checkpoints and prompt variants, deduplicate outputs, update files in real time, and print progress."
    )
    parser.add_argument("--model", "-m", required=True,
                        help="Path to the base model (e.g. mistralai/Mistral-7B-v0.1).")
    parser.add_argument("--adapter-dir", "-a", required=True,
                        help="Directory containing adapter checkpoint files and adapter_config.json.")
    parser.add_argument("--prompt", "-p", required=True, nargs='+',
                        help="One or more prompt phrases (e.g. '小雪' or '小雪 春天').")
    parser.add_argument("--start-epoch", "-e", type=int, default=1000,
                        help="Minimum epoch number of checkpoints to use (default: 1000).")
    parser.add_argument("--checkpoints", "-c", nargs='+',
                        help="Manually specify a list of .safetensors checkpoint file paths (overrides scanning adapter-dir).")
    parser.add_argument("--config-path", "-f", default=None,
                        help="Path to adapter_config.json. Defaults to <adapter-dir>/adapter_config.json.")
    parser.add_argument("--output", "-o", default=None,
                        help="Global output base name. If provided, all outputs are combined into <name>.txt and <name>.json.")
    args = parser.parse_args()

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
        for cp in args.checkpoints:
            if os.path.isfile(cp):
                epoch = extract_epoch(cp)
                checkpoint_list.append((epoch, os.path.abspath(cp)))
            else:
                print(f"Checkpoint file {cp} does not exist; skipping.", file=sys.stderr)
        checkpoint_list.sort(key=lambda x: (x[0] if x[0] is not None else x[1]))
    else:
        checkpoint_list = get_checkpoint_files(args.adapter_dir, args.start_epoch)
    if not checkpoint_list:
        print("No checkpoint files found; exiting.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(checkpoint_list)} checkpoint(s).")

    # If --output is provided, use global output files.
    if args.output:
        global_base = args.output
        global_txt = f"{global_base}.txt"
        global_json = f"{global_base}.json"
        global_unique = {}
        for phrase in args.prompt:
            process_prompt(phrase, checkpoint_list, args.model, config_path, global_txt, global_json, global_unique)
        print(f"All generations processed. Final outputs saved to: {global_txt} and {global_json}")
    else:
        # Otherwise, process each prompt separately.
        for phrase in args.prompt:
            safe_phrase = sanitize_filename(phrase)
            txt_filename = f"{safe_phrase}.txt"
            json_filename = f"{safe_phrase}_metadata.json"
            print(f"Processing prompt: {phrase} (outputs will be saved to {txt_filename} and {json_filename})")
            process_prompt(phrase, checkpoint_list, args.model, config_path, txt_filename, json_filename)
        print("All generations processed for all prompt phrases.")

if __name__ == "__main__":
    main()
