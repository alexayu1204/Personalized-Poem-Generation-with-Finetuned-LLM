# Technical Implementation Details: MLX LoRA Poetry Fine-tuning

This document provides in-depth technical information about the implementation of the MLX-based LoRA fine-tuning system for poetry generation.

## Data Processing Pipeline

### Training Data Format

The training data uses a specific instruction-following format with system, user, and assistant messages:

```
{
  "text": "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user [PROMPT]<|im_end|> <|im_start|>assistant [POEM]<|im_end|>"
}
```

- **System message**: Instructs the model to write a modern poem that is concise, accurate, and emotionally rich based on a provided theme
- **User message**: Provides the theme/prompt (typically very short, like "河" (river))
- **Assistant message**: Contains the poem response

### Data Preparation Workflow

The data preparation process in `create_train_jsonl.ipynb` involves:

1. Loading cleaned JSON data with poems
2. Converting data into the specific instruction format
3. Creating training and validation splits
4. Writing formatted data to JSONL files for MLX consumption

## LoRA Fine-tuning Configuration

### Fine-tuning Strategy

The project implements a parameter-efficient fine-tuning approach using LoRA:

- **Models used**: Qwen2.5 series (7B/14B/32B) in 4-bit or 8-bit quantization
- **Target layers**: Fine-tuning is applied to the top 12-24 transformer blocks (configurable via `num_layers`)
- **Parameter matrices**: LoRA targets specific parameter matrices in the attention and MLP blocks

### LoRA Parameters

Key LoRA hyperparameters include:

- **Rank (r)**: 4-8 (smaller for smaller datasets)
- **Alpha (α)**: 8-16 (typically 2× rank)
- **Dropout**: 0.1-0.2 (increased for small datasets to prevent overfitting)
- **Scale**: 8.0-16.0 (affects the magnitude of LoRA updates)

### Training Hyperparameters

Training configuration includes:

- **Batch size**: 2-4 (varies by model size)
- **Total iterations**: 800-1600 steps
- **Learning rate**: 1e-5 with cosine decay
- **Warmup steps**: ~5% of total iterations (80-100 steps)
- **Max sequence length**: 256 tokens

## Poem Generation System (`generate2.py`)

### Checkpoint Processing

The generation script processes checkpoints using:

```python
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
```

### Prompt Variation

To increase diversity, the system uses multiple prompt formulations:

```python
def build_prompt_variants(phrase):
    """
    Build and return a list of 5 prompt strings based on the provided prompt phrase.
    The prompt variants include slight differences (newlines or omission of closing tokens).
    """
    templates = [
        # Variation 1: base prompt with no extra newline.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}<|im_end|> <|im_start|>assistant",
        # Variation 2: one newline after the prompt phrase.
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}\\n<|im_end|> <|im_start|>assistant",
        # ... additional variations
    ]
    return [template.format(phrase=phrase) for template in templates]
```

### Generation Process

The generation is executed using:

```python
def run_generation(model, temp_adapter_dir, prompt):
    """
    Run the mlx_lm.generate command with the given model, temporary adapter directory,
    and prompt. Return the raw generated output.
    """
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--adapter-path", temp_adapter_dir,
        "--prompt", prompt
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR: Generation failed. Details: {e}\nOutput: {e.output}"
```

## Poem Evaluation (`process2.py`)

### Evaluation Strategy

The evaluation system uses OpenAI's API to assess and rank poem quality:

```python
class RankingOutput(BaseModel):
    rankings: List[int]
    top_poems: List[int]
    explanation: str
```

### Metadata Encoding

The system uses an efficient encoding system to track which checkpoint and prompt template generated each poem:

```python
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
    return base_checkpoint + (template_index - 1)
```

### Batched Evaluation

The system processes poems in batches for efficiency:

```python
def evaluate_batches_for_title(poems: List[dict], batch_size: int = 10, final_top: int = 5) -> List[dict]:
    """
    Evaluate poems in batches, then perform a final ranking on the top candidates from each batch.
    
    Args:
        poems: List of poem metadata dictionaries
        batch_size: Number of poems to evaluate in each batch
        final_top: Number of top poems to return
        
    Returns:
        List of the final_top poems, ranked by quality
    """
    # ... implementation details ...
```

## Human Evaluation UI (`tk_app_accept_overall_poems.py`)

### UI Structure

The Tkinter-based UI implements:

1. A progress tracking system showing completion status
2. Display areas for poem title and body
3. Action buttons for Accept/Edit/Skip/Delete operations
4. A saving mechanism for preserving human evaluations

### Data Processing

The UI processes poems in a flattened structure for easier navigation:

```python
def flatten_results(results):
    """
    Convert the nested list structure into a flat list for easier processing.
    Each item in the flat list will have:
    - outer_idx: Index in the outer list
    - inner_idx: Index in the inner list
    - image: Title of the poem
    - response: Body of the poem
    - accepted: Whether the user has accepted this poem
    - edited: Whether the user has edited this poem
    """
    flattened = []
    for outer_idx, inner_list in enumerate(results):
        for inner_idx, pair in enumerate(inner_list):
            flattened.append({
                'outer_idx': outer_idx,
                'inner_idx': inner_idx,
                'image': pair[0],
                'response': pair[1],
                'accepted': False,
                'edited': False
            })
    return flattened
```

### User Interaction Flow

The UI implements the following interaction flow:

1. Load and flatten poem data
2. Display poems one by one
3. Process user actions (accept/edit/skip/delete)
4. Reconstruct and save the updated data structure

```python
def save_changes(self):
    # Reconstruct the nested list based on original outer indices
    reconstructed = [[] for _ in range(self.total)]
    for poem in self.flattened_results:
        outer = poem['outer_idx']
        inner = poem['inner_idx']
        # Ensure that the inner list has enough elements
        while len(reconstructed[outer]) <= inner:
            reconstructed[outer].append(['', ''])
        reconstructed[outer][inner] = [poem['image'], poem['response']]

    # Save to JSON
    try:
        with open('updated_overall_poems.json', 'w', encoding='utf-8') as f:
            json.dump(reconstructed, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("Saved", "All changes have been saved to 'updated_overall_poems.json'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save file: {e}")
```

## Optimizations and Performance Considerations

### Memory Efficiency

1. **LoRA**: Reduces parameter count from billions to millions
2. **Quantization**: Uses 4-bit or 8-bit quantized base models
3. **Layer targeting**: Only fine-tunes the most important layers

### Training Speed

1. **MLX native**: Leverages Apple's optimized ML framework
2. **Batch size optimization**: Adjusts based on model size and available memory
3. **Sequence length limitation**: Uses 256 tokens to speed up training

### Generation Quality

1. **Checkpoint averaging**: Selects best checkpoints based on validation loss
2. **Prompt engineering**: Multiple formulations for diverse outputs
3. **Evaluation pipeline**: Automated ranking followed by human curation

## Iteration Loop

The system is designed to support an iterative improvement process:

1. Fine-tune model on initial dataset
2. Generate poems using checkpoints
3. Evaluate and rank outputs
4. Human curation via UI
5. Incorporate high-quality outputs into training data
6. Repeat the process with enriched dataset 