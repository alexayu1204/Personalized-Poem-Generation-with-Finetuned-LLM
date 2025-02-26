#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import math
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from pydantic import BaseModel
import datetime

# Check if the OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
)

# ----------------------------------------------------------------
# --- Pydantic models for structured output ---
# ----------------------------------------------------------------
# class RankingItem(BaseModel):
#     model_checkpoint: int
#     rank: int

class RankingOutput(BaseModel):
    rankings: List[int]
    top_poems: List[int]
    explanation: str

# class TopRankingOutput(BaseModel):
#     rankings: List[int]
#     top_poems: List[int]

# Helper functions to write to txt file
def apped_to_txt_file(file_path: str, prompt: str, content: str) -> None:
    if content == "":
        # This section marks the beginning of a new run in the log file for easier parsing later.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("explanation.txt", "a") as file:
            file.write("\n\n")
            file.write("==================================================\n")
            file.write(f"NEW RUN: {prompt} - {current_time}\n")
            if content != "":
                file.write(content)
            file.write("==================================================\n")
            file.write("\n")
        return
    
    if prompt == "":
        # If the prompt is empty, just write the content.
        with open(file_path, "a") as file:
            file.write(content)
            file.write("\n")
        return
    
    # Open the file in append mode and write the formatted output
    with open(file_path, "a") as file:
        file.write("--------\n")
        file.write(f"Prompt : {prompt}\n")
        file.write(f"Explanation : {content}\n")
        file.write("--------\n")

# ----------------------------------------------------------------
# --- Utility Functions for Metadata Collapsing/Expanding ---
# ----------------------------------------------------------------
def encode_checkpoint(base_checkpoint: int, template_index: int) -> int:
    """
    Given a base checkpoint (which must be a multiple of 100) and a template index (1 to 5),
    return the encoded checkpoint number.
    
    - Template 1 (base prompt): returns base_checkpoint (e.g., 1200)
    - Template 2 (adds one newline): returns base_checkpoint + 1 (e.g., 1201)
    - Template 3 (adds two newlines): returns base_checkpoint + 2 (e.g., 1202)
    - Template 4: returns base_checkpoint + 3 (e.g., 1203)
    - Template 5: returns base_checkpoint + 4 (e.g., 1204)
    """
    # if template_index < 1 or template_index > 5:
    #     raise ValueError("template_index must be between 1 and 5")
    return base_checkpoint + (template_index - 1)

def decode_checkpoint(checkpoint: int) -> Tuple[int, int]:
    """
    Given an encoded checkpoint number, return a tuple (base_checkpoint, template_index).
    
    The base_checkpoint is the largest multiple of 100 less than or equal to the checkpoint,
    and the template_index is computed as (checkpoint - base_checkpoint + 1).
    
    For example, if checkpoint = 1204, then base_checkpoint = 1200 and template_index = 5.
    """
    base = (checkpoint // 100) * 100
    template_index = checkpoint - base + 1
    if template_index < 1 or template_index > 5:
        raise ValueError("Encoded checkpoint does not represent a valid template index (must be between 1 and 5).")
    return base, template_index

# --- Functions to Get/Collapse/Expand Prompt Templates ---
def get_prompt_templates(provided: Optional[List[str]] = None, metadata: Optional[List[dict]] = None) -> List[str]:
    """
    Return a list of distinct prompt templates.
    
    If a list is provided in 'provided', return it.
    Otherwise, if metadata is provided, extract all distinct 'prompt' strings (in order of first appearance).
    """
    if provided is not None:
        return provided
    elif metadata is not None:
        templates = []
        for entry in metadata:
            prompt_text = entry.get("prompt", "").strip()
            if prompt_text and prompt_text not in templates:
                templates.append(prompt_text)
        return templates
    else:
        return []
    
def collapse_metadata(metadata: List[dict], prompt_templates: Optional[List[str]] = None) -> Tuple[List[dict], List[str]]:
    """
    For each metadata entry, remove the full "prompt" text and encode its prompt-template variant.
    
    If 'prompt_templates' is provided, use that list; otherwise, automatically extract distinct prompt strings from metadata.
    The function updates each entry's "checkpoint_number" by encoding the template index (based on the prompt text)
    and removes the "prompt" field. It also sets a flag "minimized": True.
    
    Returns the modified metadata and the list of prompt templates used.
    """
    # First, if not provided, extract templates from metadata.
    templates = get_prompt_templates(provided=prompt_templates, metadata=metadata)
    # Optionally, you might want to enforce that len(templates) <= 5.
    for entry in metadata:
        prompt_text = entry.get("prompt", "").strip()
        if prompt_text in templates:
            template_index = templates.index(prompt_text) + 1  # 1-indexed
        else:
            # If not found, assign default index 1.
            template_index = 1
        # Assume the current checkpoint_number is a multiple of 100 (or round down).
        original_cp = entry.get("checkpoint_number", 0)
        base_cp = (original_cp // 100) * 100
        entry["checkpoint_number"] = encode_checkpoint(base_cp, template_index)
        # Remove the full prompt field.
        if "prompt" in entry:
            del entry["prompt"]
        entry["minimized"] = True
    return metadata, templates

def expand_metadata(metadata: List[dict], prompt_templates: List[str]) -> List[dict]:
    """
    Given minimized metadata and a list of 5 prompt templates, reconstruct the full "prompt" text.
    
    The prompt_templates list should have 5 elements corresponding to the 5 variations.
    For each entry, decode the checkpoint number to get the template_index and then set
    the "prompt" field to prompt_templates[template_index - 1].
    
    For example, if an entry has "checkpoint_number": 1203, then decode_checkpoint(1203)
    returns (1200, 4) and the full prompt is prompt_templates[3].
    """
    for entry in metadata:
        cp = entry.get("checkpoint_number")
        if cp is None:
            continue
        base, template_index = decode_checkpoint(cp)
        # You may choose to incorporate the base value into the prompt if needed.
        entry["prompt"] = prompt_templates[template_index - 1]
        # Remove the minimized flag if desired.
        entry.pop("minimized", None)
    return metadata

# --- Standard Utility Functions ---
def load_metadata(infile: str) -> List[dict]:
    """Load metadata JSON and deduplicate entries based on 'body'."""
    try:
        with open(infile, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {infile}: {e}")
        sys.exit(1)
    unique = {}
    for entry in data:
        body = entry.get("body", "").strip()
        if body and body not in unique:
            unique[body] = entry
    return list(unique.values())

# def load_and_collapse_metadata(infile: str, prompt_templates: Optional[List[str]] = None) -> Tuple[List[dict], List[str]]:
#     """
#     Load metadata from a JSON file, deduplicate entries, and collapse each entry by encoding the prompt.
    
#     Returns a tuple: (collapsed_metadata, prompt_templates_used)
#     """
#     data = load_metadata(infile)
#     collapsed_data, templates = collapse_metadata(data, prompt_templates)
#     return collapsed_data, templates

# --- Standard Utility Functions ---
def group_by_title(metadata: List[dict]) -> Dict[str, List[dict]]:
    """Group poem metadata by their 'title' field."""
    groups = {}
    for entry in metadata:
        title = entry.get("title", "Unknown")
        groups.setdefault(title, []).append(entry)
    return groups

def group_list(lst: List, group_size: int) -> List[List]:
    """Break list into chunks of group_size."""
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]

# ----------------------------------------------------------------
# --- Structured API call with fallback ---
# ----------------------------------------------------------------
def call_chatgpt_structured_top(prompt: str, model = "o3-mini") -> RankingOutput:
    """
    Use the ChatGPT structured output API to get ranking.
    First try the beta parse method with response_format=RankingOutput.
    On failure, fall back to a regular API call and attempt to parse the output as JSON.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert poetry critic. 你是一位精通诗歌解析和信息抽取的专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format=RankingOutput,
            reasoning_effort="high",
        )
        apped_to_txt_file("explanation.txt", prompt, completion.choices[0].message.parsed.explanation)
        try:
            # print ranking list and best model checkpoint numenbr
            print("Ranking list:", completion.choices[0].message.parsed.rankings)
            print("Tops:", completion.choices[0].message.parsed.top_poems)
            print("Explanation :", completion.choices[0].message.parsed.explanation)
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Structured API call failed (attempt 0): {e}. Trying fallback using json.loads with current return.")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            print("Parsed ranking list:", parsed["rankings"])
            print("Parsed tops:", parsed["top_poems"])
            print("Parsed explanation:", parsed["explanation"])
            return RankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Structured API call failed (attempt 1): {e2}. Trying fallback using regular API call.")
        # Fallback: Try plain API call and parse output
        print("using 4o-2024-11-20 model for fallback with prompt")
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are an expert poetry critic. 你是一位精通诗歌解析和信息抽取的专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return RankingOutput.parse_obj(parsed)

    print("Failed to get valid structured output after several attempts.")
    return None

# --- Prompt Builders ---
def build_top_ranking_prompt(poems: List[dict], desired: int) -> str:
    """
    Build a ranking prompt for up to 10 poems, asking ChatGPT to rank them and return the top 'desired' ones.
    Expected schema:
      {
        "rankings": [<number>, ...],
        "top_poems": [<number>, ...],
        "explanation": "<string>"
      }
    """
    prompt = (
        "You will be given poems of a certain topic in Chinese, each identified by a unique checkpoint number. "
        "Evaluate these poems and rank them from best (1) to worst (N), focusing on their alignment with the following preferred style and qualities:\n"
        "• Modern free-verse style with introspective, existential themes\n"
        "• Vivid natural imagery\n"
        "• Exploration of life, identity, and the tension between belonging and escape\n"
        "• Philosophical or spiritual elements\n\n"
        "In your critique, consider (but do not limit yourself to) the following aspects:\n"
        "1. Tone - The poem's emotional ambience.\n"
        "2. Style - The poet's writing approach or technique.\n"
        "3. Imagery - Key visual elements that stand out.\n"
        "4. Symbolism - Underlying symbolic meanings.\n"
        "5. Themes - Central ideas explored by the poem.\n"
        "6. Poetic Techniques - Use of figurative language such as metaphor, personification.\n"
        "7. Emotional Impact - How the poem's mood or sentiment evolves.\n"
        "8. Possible Expansions - Whether the poem provokes further reflection or resonates deeply.\n\n"
        "Then, return your evaluation strictly as a JSON object matching this schema:\n\n"
        "{\n"
        "  \"rankings\": [<checkpoint_number_in_best_to_worst_order>, ...],\n"
        "  \"top_poems\": [<checkpoint_number_of_top_N_poems>, ...],\n"
        "  \"explanation\": \"<short_reason_for_your_evaluation_judgement>\"\n"
        "}\n\n"
        "Where:\n"
        "- \"rankings\" is an array listing every poem's checkpoint in descending order of quality (best poem first).\n"
        "- \"top_poems\" is an array of the checkpoint numbers corresponding to the **top N** poems (e.g., top 3 or top 5).\n"
        "- \"explanation\" is a brief, clear, and precise rationale for your overall rankings.\n\n"
        "Important: Output only valid JSON. Provide no other text beyond this JSON structure.\n\n"
        "Here are the poems:\n"
    )

    for i, poem in enumerate(poems, start=1):
        cp = poem.get("checkpoint_number", "N/A")
        body = poem.get("body", "").strip()
        prompt += f"[Checkpoint {cp}]\n{body}\n\n"
    prompt += f"Please provide your rankings and list the top {desired} poems (by checkpoint number) now."
    return prompt

# --- Evaluation Functions per Title ---
def evaluate_batches_for_title(poems: List[dict], batch_size: int = 10, final_top: int = 5) -> List[dict]:
    """
    For poems under one title, partition them into mini-batches of 'batch_size' (10).
    Then, for each batch, pick the top N poems where N = ceil(final_top / number_of_batches).
    Return the combined candidate list.
    """
    batches = group_list(poems, batch_size)
    M = len(batches)
    if M == 0:
        return []
    # Determine N such that M * N >= final_top.
    N = math.ceil(final_top / M)
    print(f"For {M} batches and desired top {final_top}, selecting top {N} from each batch.")
    candidates = []
    for i, batch in enumerate(batches, start=1):
        print(f"Evaluating mini-batch {i}/{M} with {len(batch)} poems.")
        # For each mini-batch, call API to rank and return top N.
        prompt = build_top_ranking_prompt(batch, desired=N)
        ranking_output = call_chatgpt_structured_top(prompt)
        if ranking_output:
            for cp in ranking_output.top_poems:
                # Find the corresponding poem in this batch.
                for poem in batch:
                    if str(poem.get("checkpoint_number")) == str(cp):
                        candidates.append(poem)
                        break
        else:
            print("Warning: Ranking API call failed for a mini-batch; taking first poem from the batch.")
            candidates.append(batch[0])
    return candidates

def final_ranking(candidates: List[dict], final_top: int = 5) -> List[dict]:
    """
    Given a candidate list of poems (from mini-batches), if the number is larger than final_top,
    call the ranking API on all candidates (if number is ≤ 10) or partition them into groups of 10
    and then merge them; finally, select the final_top poems.
    For simplicity, if len(candidates) <= 10, directly rank them.
    """
    if len(candidates) <= final_top:
        return candidates
    if len(candidates) <= 10:
        prompt = build_top_ranking_prompt(candidates, desired=final_top)
        ranking_output = call_chatgpt_structured_top(prompt)
        if ranking_output:
            selected = []
            for cp in ranking_output.top_poems:
                for poem in candidates:
                    if str(poem.get("checkpoint_number")) == str(cp):
                        selected.append(poem)
                        break
                if len(selected) == final_top:
                    break
            return selected
        else:
            print("Warning: Final ranking API call failed; returning first final_top candidates.")
            return candidates[:final_top]
    else:
        # If candidates > 10, further partition into mini-batches of 10 and then merge using similar logic.
        sub_batches = group_list(candidates, 10)
        sub_candidates = []
        for batch in sub_batches:
            prompt = build_top_ranking_prompt(batch, desired=min(len(batch), final_top))
            ranking_output = call_chatgpt_structured_top(prompt)
            if ranking_output:
                for cp in ranking_output.top_poems:
                    for poem in batch:
                        if str(poem.get("checkpoint_number")) == str(cp):
                            sub_candidates.append(poem)
                            break
            else:
                sub_candidates.extend(batch[:1])
        # Now sub_candidates should be ≤ len(candidates). If still > final_top, rank them directly.
        if len(sub_candidates) <= 10:
            prompt = build_top_ranking_prompt(sub_candidates, desired=final_top)
            ranking_output = call_chatgpt_structured_top(prompt)
            if ranking_output:
                final_selected = []
                for cp in ranking_output.top_poems:
                    for poem in sub_candidates:
                        if str(poem.get("checkpoint_number")) == str(cp):
                            final_selected.append(poem)
                            break
                    if len(final_selected) == final_top:
                        break
                return final_selected
            else:
                return sub_candidates[:final_top]
        else:
            # As a fallback, return the first final_top candidates.
            return sub_candidates[:final_top]

def evaluate_poems_for_title(poems: List[dict], batch_size: int = 10, final_top: int = 5) -> List[dict]:
    """
    For a given title, partition poems into batches of batch_size (10),
    from each batch select top N such that total candidates >= final_top,
    then re-rank candidates to choose final_top.
    """
    print(f"Evaluating {len(poems)} poems for title '{poems[0].get('title')}'.")
    candidates = evaluate_batches_for_title(poems, batch_size=batch_size, final_top=final_top)
    print(f"Collected {len(candidates)} candidate poems from mini-batches.")
    final_candidates = final_ranking(candidates, final_top=final_top)
    print(f"Final selected {len(final_candidates)} poems for title '{poems[0].get('title')}'.")
    return final_candidates

# --- Main Processing Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Process a JSON metadata file of generated poems (with multiple titles), deduplicate them, "
                    "and for each title, use batch ranking (sending 10 poems at a time) to select the top poems. "
                    "For each title, the final output is the best 'final_top' poems (default 5)."
    )
    parser.add_argument("--infile", "-i", required=True, help="Input JSON file containing poem metadata.")
    parser.add_argument("--outfile", "-o", default="best_poems_by_title.json",
                        help="Output JSON file to save the top poems for each title.")
    parser.add_argument("--desired", "-d", type=int, default=5,
                        help="Number of top poems to select per title (default: 5).")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of poems per mini-batch for ranking (default: 10).")
    # Optionally, you can provide a file with a list of prompt templates, or set them here.
    # If not provided, the code will extract distinct prompts automatically from the metadata.
    # PROMPT_TEMPLATES = [
    #     "Base prompt template text...",
    #     "Prompt template with one newline...",
    #     "Prompt template with two newlines...",
    #     "Prompt template variation 4...",
    #     "Prompt template variation 5..."
    # ]
    parser.add_argument("--prompt-templates", "-p", type=str, nargs="*",
                        help="Optional: Provide a list of prompt templates. If not given, they will be extracted automatically from metadata.")
    args = parser.parse_args()

    # Load and collapse metadata; if a list of prompt templates is provided via the command line, use it.
    prompt_templates_input = args.prompt_templates if args.prompt_templates and len(args.prompt_templates) > 0 else None
    data = load_metadata(args.infile)
    metadata, templates = collapse_metadata(data, prompt_templates_input)
    for template in templates:
        apped_to_txt_file("prompt_templates.txt", "", template)
    print(f"Loaded and collapsed {len(metadata)} unique poem metadata entries from {args.infile}.")
    print("Prompt templates used:")
    for idx, t in enumerate(templates, start=1):
        print(f"Template {idx}: {t}")

    groups = group_by_title(metadata)
    print(f"Found {len(groups)} distinct titles.")

    final_results = {}
    for title, poems in groups.items():
        apped_to_txt_file("explanation.txt", title, "")
        print(f"\nProcessing title: '{title}' with {len(poems)} poems.")
        top_poems = evaluate_poems_for_title(poems, batch_size=args.batch_size, final_top=args.desired)
        final_results[title] = top_poems

    try:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nTop poems by title saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()

# --- Optional: Expanding Minimized Metadata to Full Prompt Version ---
# The following code (commented out) shows how to convert the minimized metadata back to a full version.
# You need to provide a list of 5 prompt templates corresponding to the 5 variations.
#
# Example prompt templates (replace with your actual templates):
#
# PROMPT_TEMPLATES = [
#     "Base prompt template text here...",
#     "Prompt template with one newline here...",
#     "Prompt template with two newlines here...",
#     "Prompt template variation 4 text here...",
#     "Prompt template variation 5 text here..."
# ]
#
# def expand_minimized_metadata(infile: str, outfile: str, prompt_templates: List[str]) -> None:
#     with open(infile, "r", encoding="utf-8") as f:
#         minimized_data = json.load(f)
#     # For each title and each poem entry, recover the full prompt from the checkpoint number.
#     for title, poems in minimized_data.items():
#         for entry in poems:
#             cp = entry.get("checkpoint_number")
#             if cp is not None:
#                 base, template_index = decode_checkpoint(cp)
#                 entry["prompt"] = prompt_templates[template_index - 1]
#                 # Optionally remove the minimized flag:
#                 entry.pop("minimized", None)
#     with open(outfile, "w", encoding="utf-8") as f:
#         json.dump(minimized_data, f, ensure_ascii=False, indent=2)
#     print(f"Expanded metadata saved to {outfile}")
#
# # To run the expansion, uncomment and modify the following lines:
# # PROMPT_TEMPLATES = [
# #     "Base prompt template text here...",
# #     "Prompt template with one newline here...",
# #     "Prompt template with two newlines here...",
# #     "Prompt template variation 4 text here...",
# #     "Prompt template variation 5 text here..."
# # ]
# # expand_minimized_metadata("best_poems_by_title_minimized.json", "best_poems_by_title_full.json", PROMPT_TEMPLATES)