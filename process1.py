#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from typing import List, Dict
from openai import OpenAI
from pydantic import BaseModel


if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
)

# --- Pydantic models for structured output ---
# class RankingItem(BaseModel):
#     model_checkpoint: int
#     rank: int

class RankingOutput(BaseModel):
    rankings: List[int]
    best_poem: int

class TopRankingOutput(BaseModel):
    rankings: List[int]
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
def call_chatgpt_structured(prompt: str, model = "o3-mini") -> RankingOutput:
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
            # print ranking list and best model checkpoint numenbr
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
        # Fallback: Try plain API call and parse output
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

def call_chatgpt_structured_top(prompt: str) -> TopRankingOutput:
    """
    Similar to call_chatgpt_structured, but for ranking a smaller set of poems.
    Returns a TopRankingOutput object.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "Return only valid JSON matching the specified schema."},
                {"role": "user", "content": prompt}
            ],
            response_format=TopRankingOutput,
            reasoning_effort="high",
        )
        try:
            # print ranking list and best model checkpoint numenbr
            print("Ranking list:", completion.choices[0].message.parsed.rankings)
            print("Top poems:", completion.choices[0].message.parsed.top_poems)
            return completion.choices[0].message.parsed
        except Exception as e1:
            print(f"Using json .loads with current return (attempt 0): {e1}")
            content = completion.choices[0].message.content.strip()
            parsed = json.loads(content)
            print("Parsed ranking list:", parsed["rankings"])
            print("Parsed top poems:", parsed["top_poems"])
            return TopRankingOutput.parse_obj(parsed)
    except Exception as e2:
        print(f"Using regular API call (attempt 1): {e2}")
        try:
            # Fallback: Try plain API call and parse output
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

# --- Ranking Prompt Builders ---
def build_ranking_prompt(poems: List[dict]) -> str:
    """
    Build a ranking prompt for up to 10 poems.
    Instruct ChatGPT to rank these poems from best (1) to worst and return a JSON object with schema:
    {
      "rankings": [{"model_checkpoint": <number>, "rank": <number>}, ...],
      "best_poem": <number>
    }
    """
    prompt = (
        "You are an expert poetry critic. Evaluate the following poems for creativity, emotional depth, language quality, "
        "and overall impact. Each poem is identified by its model checkpoint number. Rank these poems from best (1) to worst, "
        "and return your evaluation as a JSON object exactly matching this schema:\n\n"
        "{\n  \"rankings\": [\n    {\"model_checkpoint\": <number>, \"rank\": <number>},\n    ...\n  ],\n  \"best_poem\": <number>\n}\n\n"
        "Output only valid JSON and nothing else.\n\n"
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
    Build a ranking prompt for up to 10 poems, asking ChatGPT to select the top 'desired' ones.
    Schema:
    {
      "rankings": [{"model_checkpoint": <number>, "rank": <number>}, ...],
      "top_poems": [<number>, ...]
    }
    """
    prompt = (
        "You are an expert poetry critic. Evaluate the following poems (each identified by its model checkpoint number) "
        "and rank them from best (1) to worst. Then, return your evaluation as a JSON object exactly matching this schema:\n\n"
        "{\n  \"rankings\": [\n    {\"model_checkpoint\": <number>, \"rank\": <number>},\n    ...\n  ],\n  \"top_poems\": [<number>, ...]\n}\n\n"
        "Output only valid JSON and nothing else.\n\n"
        "Here are the poems:\n"
    )
    for i, poem in enumerate(poems, start=1):
        cp = poem.get("checkpoint_number", "N/A")
        body = poem.get("body", "").strip()
        prompt += f"{i}. [Checkpoint {cp}]\n{body}\n\n"
    prompt += f"Please provide your rankings and list the top {desired} poems (by checkpoint number) now."
    return prompt

# --- Tournament Ranking Function ---
def tournament_rank_poems(poems: List[dict], max_input: int = 10, desired: int = 5) -> List[dict]:
    """
    Use a tournament ranking approach so that each API call sends at most max_input (10) poems.
    Repeatedly partition the candidate poems into batches of 10, get the best from each batch,
    and then combine the winners. Finally, when the candidates are ≤ max_input, rank them to choose the top 'desired' (e.g. 5).
    """
    candidates = poems
    round_num = 1
    while len(candidates) > max_input:
        print(f"Round {round_num}: {len(candidates)} candidates; partitioning into batches of {max_input}.")
        winners = []
        batches = group_list(candidates, max_input)
        for batch in batches:
            prompt = build_ranking_prompt(batch)
            ranking_output = call_chatgpt_structured(prompt)
            if ranking_output:
                best_cp = ranking_output.best_poem
                for poem in batch:
                    if str(poem.get("checkpoint_number")) == str(best_cp):
                        winners.append(poem)
                        break
            else:
                print("Warning: Ranking API failed for a batch; taking the first poem as winner.")
                winners.append(batch[0])
        candidates = winners
        round_num += 1

    print(f"Final tournament round: {len(candidates)} candidates (≤ {max_input}).")
    if len(candidates) <= desired:
        return candidates
    else:
        prompt = build_top_ranking_prompt(candidates, desired)
        ranking_output = call_chatgpt_structured_top(prompt)
        if ranking_output:
            top_cps = ranking_output.top_poems
            selected = []
            for cp in top_cps:
                for poem in candidates:
                    if str(poem.get("checkpoint_number")) == str(cp):
                        selected.append(poem)
                        break
                if len(selected) == desired:
                    break
            return selected
        else:
            print("Warning: Final ranking API call failed; returning first few candidates.")
            return candidates[:desired]

# --- Evaluation Function per Title ---
def evaluate_poems_for_title(poems: List[dict], desired: int = 5) -> List[dict]:
    """
    For a given title, use the tournament approach to select the top 'desired' poems.
    """
    total = len(poems)
    print(f"Total poems for title '{poems[0].get('title')}' = {total}")
    # Always use the tournament approach (which sends 10 at a time) regardless of total count.
    top_poems = tournament_rank_poems(poems, max_input=10, desired=desired)
    return top_poems

# --- Main Processing Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Process a JSON metadata file of generated poems (with multiple titles), deduplicate them, "
                    "and for each title, use a tournament ranking approach (with API calls limited to 10 poems at a time) "
                    "to select the top poems (default top 5)."
    )
    parser.add_argument("--infile", "-i", required=True, help="Input JSON file containing poem metadata.")
    parser.add_argument("--outfile", "-o", default="best_poems_by_title.json",
                        help="Output JSON file to save the top poems for each title.")
    parser.add_argument("--desired", "-d", type=int, default=5,
                        help="Number of top poems to select per title (default: 5).")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    metadata = load_metadata(args.infile)
    print(f"Loaded {len(metadata)} unique poem metadata entries from {args.infile}.")

    groups = group_by_title(metadata)
    print(f"Found {len(groups)} distinct titles.")

    final_results = {}
    for title, poems in groups.items():
        print(f"\nProcessing title: '{title}' with {len(poems)} poems.")
        top_poems = evaluate_poems_for_title(poems, desired=args.desired)
        print(f"Selected {len(top_poems)} top poems for title '{title}'.")
        final_results[title] = top_poems

    try:
        with open(args.outfile, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        print(f"\nTop poems by title saved to {args.outfile}")
    except Exception as e:
        print(f"Error saving output file: {e}")

if __name__ == "__main__":
    main()
