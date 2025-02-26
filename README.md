# Qwen (7B/14B/32B) Fine-Tuning with LoRA for Poetry Generation

A comprehensive toolkit for fine-tuning Qwen language models on custom poetry datasets using Low-Rank Adaptation (LoRA) with the MLX library on Apple Silicon. The primary goal is to create a model that can generate poetry in a specific personal style with minimal computational resources.

This project enables you to input themes or words and generate poems that reflect a distinct style. Starting with approximately 100 original poems, the models are trained via LoRA (updating only key layers) and later refined with synthetic data selected through human and AI evaluation.

## Table of Contents

- [Technical Overview](#technical-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Fine-Tuning Process](#fine-tuning-process)
- [Poem Generation](#poem-generation)
- [Adapter Management](#adapter-management)
- [Evaluation and Refinement](#evaluation-and-refinement)
- [Full Command Reference](#full-command-reference)
- [Code Implementation](#code-implementation)
- [Performance Benchmarks](#performance-benchmarks)
- [License](#license)
- [Conclusion](#conclusion)

## Technical Overview

This project implements an end-to-end pipeline for parameter-efficient fine-tuning of large language models for poetry generation:

1. **Environment Setup**: Configures the MLX library optimized for Apple Silicon with appropriate dependencies.

2. **Data Preparation**: Formats approximately 100 poems into instruction-following JSONL format with system, user, and assistant messages.

3. **LoRA Fine-Tuning**: Applies low-rank adaptation to key parameter matrices in specific transformer layers of Qwen models, enabling efficient training with minimal memory requirements.

4. **Poem Generation**: Uses fine-tuned adapters to generate poems based on themes or keywords.

5. **Evaluation**: Implements both automated evaluation (via OpenAI API) and human review of generated poems.

6. **Iterative Refinement**: Incorporates high-quality generated poems into the training dataset for subsequent fine-tuning iterations.

7. **Human-in-the-Loop**: Enables human selection and curation of generated poems via a custom UI tool.

The project leverages three key technologies:
- **MLX**: Apple's machine learning framework optimized for Apple Silicon
- **LoRA**: Parameter-efficient fine-tuning technique that trains only low-rank decomposition matrices
- **Qwen models**: High-quality transformer-based language models with 7B/14B/32B parameters

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- MLX library
- Tkinter (for UI-based evaluation)
- OpenAI API key (for automated evaluation)

## Installation

```bash
# Create and activate a conda environment
conda create -n mlx_poetry python=3.9
conda activate mlx_poetry

# Install required packages
pip install mlx tkinter numpy pydantic openai

# Clone the repository
git clone https://github.com/yourusername/mlx_lora_finetuning.git
cd mlx_lora_finetuning
```

## Project Structure

```
├── data/                      # Training data directory
│   ├── train.jsonl            # Training data in JSONL format
│   └── valid.jsonl            # Validation data in JSONL format
├── yaml/                      # Configuration files for fine-tuning
│   ├── lora1.yaml             # LoRA configuration for 7B model
│   ├── lora2.yaml             # LoRA configuration for 32B model
│   └── ...                    # Additional configuration variants
├── input/                     # Input files for generation
│   ├── prompt_templates.txt   # Templates for poem generation
│   ├── words.txt              # Theme words for poem generation
│   └── overall_poems.json     # Collected generated poems
├── adapters-poem-generator-cn/  # Directory for adapter files
├── generate2.py               # Enhanced poem generation script
├── process2.py                # Poem evaluation and ranking script
├── tk_app_accept_overall_poems.py  # Tkinter UI for human evaluation
└── create_train_jsonl.ipynb   # Notebook for data preparation
```

## Data Preparation

The training data consists of structured conversations in JSONL format. Each entry follows this pattern:

```json
{"text": "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user [THEME]<|im_end|> <|im_start|>assistant [POEM CONTENT]<|im_end|>"}
```

The `create_train_jsonl.ipynb` notebook handles:
1. Loading raw poetry data from JSON files
2. Cleaning and formatting into the required instruction-following structure
3. Creating training and validation splits
4. Writing to JSONL files compatible with the MLX library

## Fine-Tuning Process

The fine-tuning process applies LoRA to specific layers of the Qwen model, significantly reducing the number of trainable parameters.

### LoRA Configuration

Fine-tuning is configured through YAML files specifying:

```yaml
model: "mlx-community/Qwen2.5-32B-Instruct-8bit"
fine_tune_type: lora
num_layers: 16   # Apply LoRA to last 16 transformer blocks
batch_size: 4
iters: 1200      # Total training steps
lora_parameters:
  keys: [
    "self_attn.q_proj",
    "self_attn.v_proj",
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj"
  ]
  rank: 4             # Low-rank decomposition dimension
  alpha: 8            # Scaling factor (typically 2 × rank)
  dropout: 0.1        # Regularization to prevent overfitting
```

### Training Commands

Basic training with configuration file:
```bash
mlx_lm.lora -c yaml/lora5.yaml
```

Direct parameter specification:
```bash
mlx_lm.lora --model mlx-community/Qwen2.5-32B-Instruct-8bit --train \
    --data ./data --num-layers 8 --iters 1000 \
    --adapter-path 32b_adapter1
```

Extended training with custom learning rate:
```bash
mlx_lm.lora --model mlx-community/Qwen2.5-32B-Instruct-8bit --train \
    --data ./data --iters 3600 --adapter-path 32b_adapter4 \
    --learning-rate 2e-6
```

## Poem Generation

After fine-tuning, poems can be generated using the trained LoRA adapters.

### Basic Generation

```bash
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path adapter_lora2 \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 河<|im_end|> <|im_start|>assistant"
```

### Batch Generation

For generating multiple poems, the repository includes specialized scripts:

```bash
python generate2.py -p "$(cat input/words.txt)" \
    -m mlx-community/Qwen2.5-32B-Instruct-8bit \
    -a adapter_lora5 -o test_lora5_words -e 300
```

The `generate2.py` script implements:
1. Automatic checkpoint selection and management
2. Prompt template variations to encourage diversity
3. Parallel generation of multiple poems
4. Deduplication and output formatting

## Adapter Management

The MLX library offers several approaches for managing LoRA adapters:

### Checkpoint Selection

During training, checkpoints are saved periodically. Specific checkpoints can be selected for generation:

```bash
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path adapter_lora5/0001600_adapters.safetensors \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 山<|im_end|> <|im_start|>assistant"
```

### Adapter Comparison

Different adapters can be compared against the same prompt:

```bash
# First adapter
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path 32b_adapter1 \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 雨<|im_end|> <|im_start|>assistant"

# Second adapter
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path 32b_adapter4 \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 雨<|im_end|> <|im_start|>assistant"
```

## Evaluation and Refinement

The project implements a two-stage evaluation pipeline:

### Automated Evaluation

The `process2.py` script utilizes the OpenAI API to evaluate and rank generated poems:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run the evaluation script
python process2.py -i generated_poems.json -o ranked_poems.json
```

This evaluation leverages external models for quality assessment, applying metrics related to:
- Adherence to the specified theme
- Poetic quality and creativity
- Language fluency and coherence
- Emotional resonance

### Human-in-the-Loop Evaluation

A Tkinter-based UI facilitates human review of generated poems:

```bash
# Launch the Tkinter UI
python tk_app_accept_overall_poems.py
```

The UI enables:
- Sequential viewing of poems
- Acceptance or rejection decisions
- Editing capabilities for minor adjustments
- Progress tracking through the evaluation set
- Persistent storage of human judgments

### Iterative Refinement

Selected high-quality poems are incorporated into the training dataset:

```bash
# Convert selected poems to training format
python utils/convert_selections.py -i updated_overall_poems.json -o data/additional_poems.jsonl

# Combine with original training data
cat data/train.jsonl data/additional_poems.jsonl > data/train_enriched.jsonl

# Fine-tune with enriched dataset
mlx_lm.lora -c yaml/lora5.yaml
```

This iterative process enables continuous improvement of the model's stylistic understanding and generation capabilities.

## Full Command Reference

### Environment Setup

```bash
# Create and activate the conda environment
conda create -n mlx_poetry python=3.9
conda activate mlx_poetry

# Check command history
cat ~/.bash_history | grep mlx_lm

# View help for LoRA training
mlx_lm.lora --help
```

### Training Commands

```bash
# Initial LoRA training (7B model)
mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-bf16 --train \
    --data ./data --num-layers 8 --iters 1000 \
    --adapter-path adapters

# Training with custom batch size (32B model)
mlx_lm.lora --model mlx-community/Qwen2.5-32B-Instruct-8bit --train \
    --data ./data --iters 1000 \
    --adapter-path 32b_adapter2 --batch-size 8

# Extended training with custom learning rate
mlx_lm.lora --model mlx-community/Qwen2.5-32B-Instruct-8bit --train \
    --data ./data --iters 3600 \
    --adapter-path 32b_adapter4 --learning-rate 2e-6

# Training with 4-bit quantization
mlx_lm.lora --model mlx-community/Qwen2.5-32B-Instruct-4bit --train \
    --data ./data --iters 1200 \
    --adapter-path 32b_adapter_4bit --num-layers 16

# Training with configuration files
mlx_lm.lora -c yaml/lora1.yaml
mlx_lm.lora -c yaml/lora2.yaml
mlx_lm.lora -c yaml/lora3.yaml
mlx_lm.lora -c yaml/lora4.yaml
mlx_lm.lora -c yaml/lora5.yaml
```

### Generation Commands

```bash
# View generation help
mlx_lm.generate -h

# Generate with adapter directory
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path adapter_lora2 \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 河<|im_end|> <|im_start|>assistant"

# Generate with specific checkpoint
mlx_lm.generate --model mlx-community/Qwen2.5-32B-Instruct-8bit \
    --adapter-path adapter_lora4/0001200_adapters.safetensors \
    --prompt "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> <|im_start|>user 河<|im_end|> <|im_start|>assistant"

# Batch generation with words from file
python generate2.py -m mlx-community/Qwen2.5-32B-Instruct-8bit \
    -a adapter_lora2 -o generated_poems -e 100 \
    -p "$(cat input/words.txt)"

# Generate with 4-bit model and specified epoch range
python generate2.py -m mlx-community/Qwen2.5-32B-Instruct-4bit \
    -a adapters6 -o generate_poems_1 --start-epoch 3000 \
    -p "$(cat input/words.txt)"

# Large-scale generation (300 poems)
python generate2.py -p "$(cat input/words.txt)" \
    -m mlx-community/Qwen2.5-32B-Instruct-8bit \
    -a adapter_lora5 -o test_lora5_words -e 300
```

### Evaluation Commands

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run evaluation script
python process2.py -i generated_poems.json -o ranked_poems.json

# Launch human evaluation UI
python tk_app_accept_overall_poems.py

# Convert selected poems to training format
python utils/convert_selections.py -i updated_overall_poems.json -o data/additional_poems.jsonl

# Combine with original training data
cat data/train.jsonl data/additional_poems.jsonl > data/train_enriched.jsonl
```

## Code Implementation

### Prompt Template Handling

The following code from `generate2.py` demonstrates how prompt variants are constructed:

```python
def build_prompt_variants(phrase):
    """
    Build prompt variations to encourage diverse outputs.
    """
    templates = [
        # Standard prompt
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}<|im_end|> <|im_start|>assistant",
        
        # Prompt with newline
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}\\n<|im_end|> <|im_start|>assistant",
    ]
    return [template.format(phrase=phrase) for template in templates]
```

### Generation Process

The generation process is implemented as follows:

```python
def run_generation(model, temp_adapter_dir, prompt):
    """
    Execute model generation with specified adapter.
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

### Checkpoint Management

The code effectively handles adapter checkpoint selection:

```python
def get_checkpoint_files(adapter_dir, epochs):
    """
    Get available checkpoint files sorted by epoch.
    """
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(adapter_dir, "*_adapters.safetensors"))
    
    # Extract epoch numbers and sort
    checkpoint_data = []
    for filepath in checkpoint_files:
        filename = os.path.basename(filepath)
        match = re.match(r"(\d+)_adapters\.safetensors", filename)
        if match:
            epoch = int(match.group(1))
            checkpoint_data.append((epoch, filepath))
    
    # Sort by epoch and select specified number
    checkpoint_data.sort(key=lambda x: x[0])
    selected_checkpoints = checkpoint_data[:epochs] if epochs > 0 else checkpoint_data
    
    return selected_checkpoints
```

## Performance Benchmarks

- **Training Efficiency**:
  - Fine-tuning the 7B model requires approximately 800-1200 epochs for optimal results
  - 32B model training with 8-bit quantization takes 3-4 hours on M1 Max/M2 Max
  - 4-bit quantization further reduces memory requirements for 32B model

- **Generation Speed**:
  - Single poem generation: 5-10 seconds on M1 Max
  - Batch generation: ~100 poems per hour

- **Memory Usage**:
  - 7B model: ~10GB with full precision, ~6GB with 8-bit quantization
  - 32B model: ~24GB with 8-bit quantization, ~14GB with 4-bit quantization

- **Quality Metrics**:
  - After iterative refinement, ~40% of generated poems require no human editing
  - ~30% require minor adjustments
  - ~30% are discarded

## License

This project is released under the MIT License. See the LICENSE file for details.

## Conclusion

This project demonstrates an efficient pipeline for fine-tuning large language models on small poetry datasets using parameter-efficient techniques. By applying LoRA to update only key parameter matrices in specific layers, the approach enables training on consumer hardware (Apple Silicon Macs) while producing high-quality poetry outputs.

The iterative refinement process, combining automated evaluation and human curation, creates a virtuous cycle that progressively improves the model's stylistic understanding and generation capabilities. The result is a custom poetry generation system that captures specific artistic sensibilities with minimal data and computational resources.

---

By following the commands and workflows detailed above, you can reproduce the entire pipeline—from fine-tuning Qwen models using LoRA on Apple Silicon, to generating and evaluating new poems, and iteratively improving model output through human feedback. 