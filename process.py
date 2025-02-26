#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from typing import List, Dict
from pydantic import BaseModel
from openai import OpenAI

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
)

# --- Pydantic models for structured output ---
class RankingItem(BaseModel):
    model_checkpoint: int
    rank: int

class RankingOutput(BaseModel):
    rankings: List[RankingItem]
    best_poem: int

class TopRankingOutput(BaseModel):
    rankings: List[RankingItem]
    top_poems: List[int]

# --- Utility Functions ---

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

def group_by_title(metadata: List[dict]) -> Dict[str, List[dict]]:
    """Group the poem metadata by their 'title' field."""
    groups = {}
    for entry in metadata:
        title = entry.get("title", "Unknown")
        groups.setdefault(title, []).append(entry)
    return groups

def group_list(lst: List, group_size: int) -> List[List]:
    """Break list into chunks of group_size."""
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]

# --- Structured API call with fallback ---
def call_chatgpt_structured(prompt: str) -> RankingOutput:
    """
    Use the ChatGPT structured output API to get ranking.
    First try the beta parse method with response_format=RankingOutput.
    On failure, fall back to a regular API call and attempt to parse the output as JSON.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=RankingOutput,
        )
        try:
            return completion.choices[0].message.parsed
        except Exception as e:
            print(f"Structured API call failed (attempt 0): {e}. Trying fallback using json.loads with current return.")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            # Validate with Pydantic
            return RankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Structured API call failed (attempt 1): {e2}. Trying fallback using regular API call.")
        # Fallback: Try plain API call and parse output
        response = client.completions.create(
            model="gpt-4o-2024-08-06",
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

def call_chatgpt_structured_top(prompt: str) -> TopRankingOutput:
    """
    Similar to call_chatgpt_structured, but for ranking a smaller set of poems.
    Returns a TopRankingOutput object.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=TopRankingOutput,
        )
        try:
            return completion.choices[0].message.parsed
        except Exception as e1:
            print(f"Using json .loads with current return (attempt 0): {e1}")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            # Validate with Pydantic
            return TopRankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Using regular API call (attempt 1): {e2}")
        try:
            response = client.completions.create(
                model="gpt-4o-2024-08-06",
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

# --- Ranking Prompt Builders ---
def build_ranking_prompt(poems: List[dict]) -> str:
    """
    Build a ranking prompt for a batch of poems.
    Each poem is identified by its model checkpoint number.
    Instruct ChatGPT to rank the poems from best (1) to worst and return a JSON object
    matching the following schema:
    
    {
      "rankings": [
         {"model_checkpoint": <number>, "rank": <number>},
         ...
      ],
      "best_poem": <number>
    }
    
    Output only valid JSON.
    """
    prompt = (
        "You are an expert poetry critic. Evaluate the following poems for creativity, emotional depth, language quality, "
        "and overall impact. Each poem is identified by its model checkpoint number. Rank these poems from best (1) "
        "to worst, and return your evaluation as a JSON object exactly matching this schema:\n\n"
        "{\n  \"rankings\": [\n    {\"model_checkpoint\": <number>, \"rank\": <number>},\n    ...\n  ],\n  \"best_poem\": <number>\n}\n\n"
        "Do not include any extra text. Output only valid JSON.\n\n"
        "Here are the poems:\n"
    )
    for i, poem in enumerate(poems, start=1):
        cp = poem.get("checkpoint_number", "N/A")
        body = poem.get("body", "").strip()
        prompt += f"{i}. [Checkpoint {cp}]\n{body}\n\n"
    prompt += "Please provide your rankings now."
    return prompt

def build_top_ranking_prompt(poems: List[dict], desired: int) -> str:
    """
    Build a ranking prompt for a set of best poems.
    Instruct ChatGPT to rank these poems and return the top 'desired' ones.
    Schema:
    
    {
      "rankings": [
         {"model_checkpoint": <number>, "rank": <number>},
         ...
      ],
      "top_poems": [<number>, ...]   // a list of model checkpoint numbers in ranked order, limited to top 'desired'
    }
    
    Output only valid JSON.
    """
    prompt = (
        "You are an expert poetry critic. Evaluate the following top poems (each identified by its model checkpoint number) "
        "and rank them from best (1) to worst. Return your evaluation as a JSON object exactly matching this schema:\n\n"
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

# --- Evaluation Functions ---
def evaluate_batches_for_title(poems: List[dict], group_size: int = 10) -> List[dict]:
    """
    For a list of poems of the same title, break them into batches of 'group_size',
    evaluate each batch to select the best poem, then return the list of best poems.
    """
    batches = group_list(poems, group_size)
    best_from_batches = []
    for i, batch in enumerate(batches, start=1):
        print(f"Evaluating batch {i}/{len(batches)} with {len(batch)} poems for title '{poems[0].get('title')}'.")
        prompt = build_ranking_prompt(batch)
        ranking_output = call_chatgpt_structured(prompt)
        if ranking_output is not None:
            best_cp = ranking_output.best_poem
            # Find the poem with this checkpoint number
            for poem in batch:
                if str(poem.get("checkpoint_number")) == str(best_cp):
                    best_from_batches.append(poem)
                    break
        else:
            print("Warning: Failed to evaluate batch; skipping this batch.")
    return best_from_batches

def select_top_for_title(best_poems: List[dict], desired: int = 5) -> List[dict]:
    """
    From a list of best poems (from batches) for a given title,
    if there are more than 'desired', rank them to select the top 'desired' poems.
    """
    if len(best_poems) <= desired:
        return best_poems
    prompt = build_top_ranking_prompt(best_poems, desired)
    ranking_output = call_chatgpt_structured_top(prompt)
    if ranking_output is None:
        print("Warning: Failed to rank top poems; returning first ones.")
        return best_poems[:desired]
    top_cps = ranking_output.top_poems  # expected list of checkpoint numbers
    selected = []
    for cp in top_cps:
        for poem in best_poems:
            if str(poem.get("checkpoint_number")) == str(cp):
                selected.append(poem)
                break
        if len(selected) == desired:
            break
    return selected

# --- Main processing function ---
def main():
    parser = argparse.ArgumentParser(
        description="Process a JSON metadata file of generated poems (possibly with multiple titles), "
                    "deduplicate them, and for each title, use ChatGPT's structured API to evaluate and rank "
                    "the poems in batches. Finally, for each title, select the top 5 poems and save the results."
    )
    parser.add_argument("--infile", "-i", required=True, help="Input JSON file containing poem metadata.")
    parser.add_argument("--outfile", "-o", default="best_poems_by_title.json",
                        help="Output JSON file to save the top poems for each title.")
    parser.add_argument("--group-size", "-g", type=int, default=10,
                        help="Number of poems per group for evaluation (default: 10).")
    args = parser.parse_args()

    metadata = load_metadata(args.infile)
    print(f"Loaded {len(metadata)} unique poem metadata entries from {args.infile}.")

    # Group by title.
    groups = group_by_title(metadata)
    print(f"Found {len(groups)} distinct titles.")

    final_results = {}
    for title, poems in groups.items():
        print(f"\nProcessing title: '{title}' with {len(poems)} poems.")
        # Evaluate in batches for this title.
        best_batches = evaluate_batches_for_title(poems, group_size=args.group_size)
        print(f"Collected {len(best_batches)} best poems from batches for title '{title}'.")
        # If more than 5, rank them to select top 5.
        top_poems = select_top_for_title(best_batches, desired=5)
        print(f"Selected {len(top_poems)} top poems for title '{title}'.")
        final_results[title] = top_poems

    # Save final results.
    try:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nTop poems by title saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()
