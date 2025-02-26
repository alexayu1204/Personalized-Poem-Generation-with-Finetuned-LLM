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

# --- Set up OpenAI client ---
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# --- Pydantic models for structured output ---
class RankingItem(BaseModel):
    model_checkpoint: int
    rank: int

class RankingOutput(BaseModel):
    rankings: List[int]
    best_poem: int

class TopRankingOutput(BaseModel):
    rankings: List[int]
    top_poems: List[int]

# --- Utility Functions for Checkpoint Encoding/Decoding ---
def encode_checkpoint(base_checkpoint: int, template_index: int) -> int:
    """
    Given a base checkpoint (which must be a multiple of 100) and a template index (1 to 5),
    return the encoded checkpoint number.
    
    - Template 1: returns base_checkpoint (e.g., 1200).
    - Template 2: returns base_checkpoint + 1 (e.g., 1201).
    - Template 3: returns base_checkpoint + 2 (e.g., 1202).
    - Template 4: returns base_checkpoint + 3 (e.g., 1203).
    - Template 5: returns base_checkpoint + 4 (e.g., 1204).
    """
    if template_index < 1 or template_index > 5:
        raise ValueError("template_index must be between 1 and 5")
    return base_checkpoint + (template_index - 1)

def decode_checkpoint(checkpoint: int) -> Tuple[int, int]:
    """
    Given an encoded checkpoint number, return a tuple (base_checkpoint, template_index).
    
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

def load_and_collapse_metadata(infile: str, prompt_templates: Optional[List[str]] = None) -> Tuple[List[dict], List[str]]:
    """
    Load metadata from a JSON file, deduplicate entries, and collapse each entry by encoding the prompt.
    
    Returns a tuple: (collapsed_metadata, prompt_templates_used)
    """
    data = load_metadata(infile)
    collapsed_data, templates = collapse_metadata(data, prompt_templates)
    return collapsed_data, templates

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

# --- Structured API call with fallback ---
def call_chatgpt_structured_top(prompt: str, model: str = "o3-mini") -> RankingOutput:
    """
    Use the ChatGPT structured output API to get ranking.
    First try the beta parse method with response_format=RankingOutput.
    On failure, fall back to a regular API call and attempt to parse the output as JSON.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=RankingOutput,
            reasoning_effort="high",
        )
        try:
            print("Ranking list:", completion.choices[0].message.parsed.rankings)
            print("Best model checkpoint number:", completion.choices[0].message.parsed.best_poem)
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Structured API call failed (attempt 0): {e}. Trying fallback using json.loads with current return.")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            print("Parsed ranking list:", parsed["rankings"])
            print("Parsed best model checkpoint number:", parsed["best_poem"])
            return RankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Structured API call failed (attempt 1): {e2}. Trying fallback using regular API call.")
        print("using 4o-2024-11-20 model for fallback with prompt")
        print(prompt)
        response = client.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        return RankingOutput.parse_obj(parsed)
    print("Failed to get valid structured output after several attempts.")
    return None

def call_chatgpt_structured_top_fallback(prompt: str) -> TopRankingOutput:
    """
    Similar to call_chatgpt_structured_top, but for ranking a smaller set of poems.
    Returns a TopRankingOutput object.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="o3-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=TopRankingOutput,
            reasoning_effort="high",
        )
        try:
            print("Ranking list:", completion.choices[0].message.parsed.rankings)
            print("Top poems:", completion.choices[0].message.parsed.top_poems)
            return completion.choices[0].message.parsed
        except Exception as e1:
            print(f"Using json.loads with current return (attempt 0): {e1}")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            print("Parsed ranking list:", parsed["rankings"])
            print("Parsed top poems:", parsed["top_poems"])
            return TopRankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Using regular API call (attempt 1): {e2}")
        try:
            print("using 4o-2024-11-20 model for fallback with prompt")
            print(prompt)
            response = client.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            return TopRankingOutput.parse_obj(parsed)
        except Exception as e3:
            print(f"Failed to get valid top ranking output after several attempts: {e3}")
            return None
    print("Failed to get valid top ranking output after several attempts.")
    return None

# --- Prompt Builder ---
def build_top_ranking_prompt(poems: List[dict], desired: int) -> str:
    """
    Build a ranking prompt for up to 10 poems, asking ChatGPT to rank them and return the top 'desired' ones.
    Expected schema:
      {
        "rankings": [{"model_checkpoint": <number>, "rank": <number>}, ...],
        "top_poems": [<number>, ...]
      }
    """
    prompt = (
        "You are an expert poetry critic. Evaluate the following poems (each identified by its model checkpoint number) "
        "and rank them from best (1) to worst. Then, return your evaluation as a JSON object exactly matching this schema:\n\n"
        "{\n  \"rankings\": [\n    {\"model_checkpoint\": <number>, \"rank\": <number>},\n    ...\n  ],\n  \"top_poems\": [<number>, ...]\n}\n\n"
        "Do not include any extra text. Output only valid JSON.\n\n"
        "Here are the poems:\n"
    )
    for i, poem in enumerate(poems, start=1):
        cp = poem.get("checkpoint_number", "N/A")
        body = poem.get("body", "").strip()
        prompt += f"{i}. [Checkpoint {cp}]\n{body}\n\n"
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
    N = math.ceil(final_top / M)
    print(f"For {M} batches and desired top {final_top}, selecting top {N} from each batch.")
    candidates = []
    for i, batch in enumerate(batches, start=1):
        print(f"Evaluating mini-batch {i}/{M} with {len(batch)} poems.")
        prompt = build_top_ranking_prompt(batch, desired=N)
        ranking_output = call_chatgpt_structured_top_fallback(prompt)
        if ranking_output:
            for cp in ranking_output.top_poems:
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
    call the ranking API on all candidates (if number is ≤ 10, directly; if more, partition them) 
    and then select the final_top poems.
    """
    if len(candidates) <= final_top:
        return candidates
    if len(candidates) <= 10:
        prompt = build_top_ranking_prompt(candidates, desired=final_top)
        ranking_output = call_chatgpt_structured_top_fallback(prompt)
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
        sub_batches = group_list(candidates, 10)
        sub_candidates = []
        for batch in sub_batches:
            prompt = build_top_ranking_prompt(batch, desired=min(len(batch), final_top))
            ranking_output = call_chatgpt_structured_top_fallback(prompt)
            if ranking_output:
                for cp in ranking_output.top_poems:
                    for poem in batch:
                        if str(poem.get("checkpoint_number")) == str(cp):
                            sub_candidates.append(poem)
                            break
            else:
                sub_candidates.extend(batch[:1])
        if len(sub_candidates) <= 10:
            prompt = build_top_ranking_prompt(sub_candidates, desired=final_top)
            ranking_output = call_chatgpt_structured_top_fallback(prompt)
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
            return sub_candidates[:final_top]

def evaluate_poems_for_title(poems: List[dict], batch_size: int = 10, final_top: int = 5) -> List[dict]:
    """
    For a given title, partition poems into batches of batch_size,
    select top N from each batch (where N is computed so that total candidates >= final_top),
    then re-rank the combined candidates to choose final_top.
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
                    "collapse the full prompt into a minimized version (using encoded checkpoint numbers), and for each title, "
                    "use batch ranking (sending 10 poems at a time) to select the top poems. "
                    "The final output for each title is the best 'final_top' poems (default 5) in minimized form."
    )
    parser.add_argument("--infile", "-i", required=True, help="Input JSON file containing poem metadata.")
    parser.add_argument("--outfile", "-o", default="best_poems_by_title_minimized.json",
                        help="Output JSON file to save the top poems for each title in minimized form.")
    parser.add_argument("--desired", "-d", type=int, default=5,
                        help="Number of top poems to select per title (default: 5).")
    parser.add_argument("--batch-size", "-b", type=int, default=10,
                        help="Number of poems per mini-batch for ranking (default: 10).")
    # Optionally, you can provide a file with a list of prompt templates, or set them here.
    # If not provided, the code will extract distinct prompts automatically from the metadata.
    # For example, to use a fixed list:
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

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    # Load and collapse metadata; if a list of prompt templates is provided via the command line, use it.
    prompt_templates_input = args.prompt_templates if args.prompt_templates and len(args.prompt_templates) > 0 else None
    metadata, templates = load_and_collapse_metadata(args.infile, prompt_templates=prompt_templates_input)
    print(f"Loaded and collapsed {len(metadata)} unique poem metadata entries from {args.infile}.")
    print("Prompt templates used:")
    for idx, t in enumerate(templates, start=1):
        print(f"Template {idx}: {t}")

    groups = group_by_title(metadata)
    print(f"Found {len(groups)} distinct titles.")

    final_results = {}
    for title, poems in groups.items():
        print(f"\nProcessing title: '{title}' with {len(poems)} poems.")
        top_poems = evaluate_poems_for_title(poems, batch_size=args.batch_size, final_top=args.desired)
        final_results[title] = top_poems

    try:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nMinimized top poems by title saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()

# --- Optional: Expanding Minimized Metadata to Full Prompt Version ---
# The following commented-out code shows how to convert minimized metadata back to full metadata.
# You must supply a list of 5 prompt templates (either the same you provided or the ones extracted).
#
# To run the expansion, uncomment and set PROMPT_TEMPLATES:
# PROMPT_TEMPLATES = [
#     "Base prompt template text...",
#     "Prompt template with one newline...",
#     "Prompt template with two newlines...",
#     "Prompt template variation 4...",
#     "Prompt template variation 5..."
# ]
# expand_minimized_metadata("best_poems_by_title_minimized.json", "best_poems_by_title_full.json", PROMPT_TEMPLATES)