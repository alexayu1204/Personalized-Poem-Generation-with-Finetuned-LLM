# Personalized Poem Generation with Fine-Tuned Large Language Models: A Technical Deep Dive

## 1. Model Selection and Architecture

### Qwen 2.5 Foundation Model

```python
# From yaml/lora2.yaml - Key configuration for the Qwen model
model: "mlx-community/Qwen2.5-7B-Instruct-8bit"  # 8-bit quantization for memory efficiency
train: true
fine_tune_type: lora
data: "./data"
num_layers: 12   # Apply LoRA to last 12 transformer blocks only
```

**Explanation:** This configuration targets Qwen 2.5 (specialized for Chinese text) with advanced parameter-efficient fine-tuning. The implementation selectively applies Low-Rank Adaptation (LoRA) to only the last 12 layers and specific parameter matrices, focusing on components most responsible for stylistic elements. The lower rank (4) and higher dropout (0.2) are specifically calibrated for small-data scenarios to prevent overfitting and memorization of training examples.

The project deliberately uses Qwen 2.5 (7B/14B/32B parameter variants) over other options for its superior performance on Chinese text. Key architectural decisions include:

- **Quantization Strategy**: 8-bit and 4-bit quantized variants to enable training on consumer hardware
- **Layer-Selective Approach**: LoRA applied only to the last 12-16 layers, targeting the most style-specific representations
- **Instruction Format**: Specialized instruction-tuning format to leverage Qwen's existing capabilities

## 2. Data Preprocessing Pipeline

### Instruction Format Design

```python
# From preprocess/create_train_jsonl.py - Specialized data formatting
for poem in data:
    poems.append({
        'text': '<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> ' + 
                '<|im_start|>user '+ poem['image'] +'<|im_end|> ' + 
                '<|im_start|>assistant '+ poem['response'] + '<|im_end|>'
    })

# Save formatted data to JSONL files
with open("./data/train.jsonl", "w", encoding="utf-8") as f:
    for line in poems:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
```

**Explanation:** The data preprocessing pipeline preserves critical poetic elements through a specialized instruction format. The system message directs the model to write modern poetry that is concise and emotionally rich. The format maintains line breaks, punctuation patterns, and structural elements essential for Chinese poetry. Notably, each poem uses a minimal prompt (often a single word like "河" for "river") to focus the model on learning stylistic patterns rather than thematic associations.

The preprocessing pipeline preserves critical poetic elements:

- **Line Breaks**: Maintained through careful text processing
- **Punctuation**: Chinese-specific punctuation preserved to maintain rhythm and flow
- **Structural Tokens**: Special `<|im_start|>` and `<|im_end|>` tokens properly position content
- **Minimal Prompts**: Single word/phrase prompts (e.g., "河" - river) enable focused style learning

## 3. LoRA Fine-tuning Configuration

## Hyperparameter Optimization

```python
# yaml/lora2.yaml - Learning rate schedule and batch configuration for small datasets
batch_size: 4        # Small batch to prevent overgeneralization
iters: 1000          # Multiple epochs over ~100 poem dataset
val_batches: 25      # Frequent validation
steps_per_report: 20 # Regular progress updates
steps_per_eval: 100  # Checkpoint evaluation frequency

# Learning rate schedule calibrated for minimal data
learning_rate: 1e-5  # Conservative peak learning rate

lr_schedule:
  name: cosine_decay  # Smooth decay pattern for stable convergence
  warmup: 50          # ~5% of total steps for initialization stability
  warmup_init: 1e-7   # Very small initial learning rate
  arguments: [1e-5, 1000, 1e-7]  # [peak_lr, total_steps, final_lr]
```

**Explanation:** This hyperparameter configuration is specifically calibrated for small-data fine-tuning scenarios. The implementation uses a conservative learning rate (1e-5) with an extended warmup period (50 steps) to stabilize initial training. The cosine decay schedule creates a smooth learning trajectory to prevent overfitting. Small batch sizes (4) help maintain diversity in gradient updates, while frequent evaluation checkpoints (every 100 steps) enable careful monitoring of model convergence and potential overfitting.

### Parameter-Efficient Adaptation

```python
# From yaml/lora2.yaml - Targeted LoRA configuration for small datasets
lora_parameters:
  # Strategic matrix targeting for maximum style transfer with minimal parameters
  keys: [
    "self_attn.q_proj",  # Attention query projection (style focus)
    "self_attn.v_proj",  # Attention value projection (content representation)
    "mlp.down_proj",     # Downward projection in feed-forward network
    "mlp.gate_proj",     # Gating mechanism (controls information flow)
    "mlp.up_proj"        # Upward projection in feed-forward network
  ]
  rank: 4             # Low rank for data efficiency (reduces overfitting)
  alpha: 8            # Scaling factor (typically 2× rank)
  dropout: 0.2        # Higher dropout for regularization with small data
  scale: 8.0          # Update magnitude control
```

The project implements careful hyperparameter optimization for small-data scenarios:

- **Matrix Selection**: Detailed analysis of which parameter matrices most effectively capture poetic style
- **Rank Reduction**: Smaller rank (4) than typical (8-16) to reduce parameters and prevent overfitting
- **Increased Dropout**: Higher dropout (0.2) than standard (0.05-0.1) specifically for small training sets
- **Controlled Learning**: Conservative learning rates with extended warm-up periods

### Learning Rate Schedule

```python
# From yaml/lora2.yaml - Custom learning rate schedule
learning_rate: 1e-5   # Conservative peak learning rate

lr_schedule:
  name: cosine_decay
  warmup: 50          # ~5% of total steps for stability
  warmup_init: 1e-7   # Very small initial learning rate
  arguments: [1e-5, 1000, 1e-7]  # [peak_lr, total_steps, final_lr]
```

## 4. Training Implementation - LoRA Fine-tuning

### MLX Framework Integration

```bash
# Training command executing fine-tuning with MLX framework using mlx_lm.lora for 7B model
# This implementation leverages Apple Silicon's unified memory architecture
mlx_lm.lora --model mlx-community/Qwen2.5-7B-Instruct-8bit --train \
    --data ./data --num-layers 16 --iters 1200 \
    --adapter-path 7b_adapter_lora --learning-rate 1e-5
```

The training implementation utilizes Apple's MLX framework:

- **Memory Optimization**: Uses Apple Silicon's unified memory architecture for efficient training
- **Checkpoint Management**: Saves intermediate checkpoints for evaluation (every 100-200 steps)
- **Gradient Tracking**: Implements careful gradient analysis to detect overfitting
- **Batch Size Adjustment**: Dynamically adjusts batch size based on available memory

## 5. Prompt Engineering and Generation

### Template Diversification

```python
# From generate2.py - Prompt diversification strategy for varied poem inputs/outputs
def build_prompt_variants(phrase):
    """
    Create multiple prompt formulations to generate diverse poetic expressions
    and to encourage diverse outputs while maintaining stylistic consistency.
    """
    templates = [
        # Base template - standard instruction format
        "<|im_start|>system 用户会给出一个主题，请按照给定的主题，切实准确简洁且情感丰富地写一首现代诗<|im_end|> " +
        "<|im_start|>user {phrase}<|im_end|> <|im_start|>assistant",
        
        # Variant prompt after theme for different rhythm
        # Variant with newlines for stronger pause before generation
        # Variants with modified token boundaries for different completion patterns
        # ...
    ]
    return [template.format(phrase=phrase) for template in templates]
```

**Explanation:** The implementation uses systematic prompt engineering to generate diverse stylistic variations. Each theme is processed through five different template formulations that strategically vary newlines and token boundaries. This approach produces subtle rhythmic differences and structural patterns while maintaining thematic consistency. The generation metadata is carefully tracked with an encoded checkpoint system that embeds template information into checkpoint numbers, enabling precise evaluation of which prompt formulations produce the highest quality outputs.

The generation implementation includes sophisticated prompt engineering:

- **Five Template Variants**: Each creates subtle differences in poem style and structure
- **Newline Control**: Strategic placement of newlines affects rhythm and pacing
- **Token Boundary Variation**: Manipulation of closing tokens changes completion patterns
- **Template Metadata**: Each variant encoded in checkpoint numbers for tracking (e.g., 1200+0, 1200+1)

### Checkpoint Management

```python
# From generate2.py - Checkpoint identification and processing
def get_checkpoint_files(adapter_dir, start_epoch):
    """
    Scan adapter_dir for checkpoint files and return a sorted list
    of tuples (epoch, full_path) for those meeting the specified criteria.
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

**Explanation:** LoRA adapters are saved at regular intervals, with the checkpoint management system tracking training progress and enabling systematic evaluation of different training stages. Runtime checkpoint management for adapter loading is an efficient way of saving the tuning result in different period of training time that will encourage diverse output in generation through different checkpoints.



```python
# Generate poems with command-line tool, tracking generation metadata
def run_generation(model, temp_adapter_dir, prompt):
    """Execute generation with loaded adapter"""
    cmd = [
        "mlx_lm.generate",
        "--model", model,
        "--adapter-path", temp_adapter_dir,
        "--prompt", prompt
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout
```

**Explanation:** The implementation uses systematic prompt engineering to generate diverse stylistic variations. Each theme is processed through five different template formulations that strategically vary newlines and token boundaries. This approach produces subtle rhythmic differences and structural patterns while maintaining thematic consistency. The generation metadata is carefully tracked with an encoded checkpoint system that embeds template information into checkpoint numbers, enabling precise evaluation of which prompt formulations produce the highest quality outputs.

## 

## 6. Evaluation Pipeline

### Pydantic-Based Structured Evaluation

```python
# From process3.py - Structured output models for poem evaluation
class RankingOutput(BaseModel):
    rankings: List[int]
    best_poem: int

class TopRankingOutput(BaseModel):
    rankings: List[int]
    top_poems: List[int]
```

### GPT-4 Evaluation API Integration

Evaluation System with LLM and Human Feedback

```python
# From process3.py - Structured LLM evaluation with detailed literary criteria
def build_top_ranking_prompt(poems: List[dict], desired: int) -> str:
    """
    Build comprehensive evaluation prompt with specific literary evaluation criteria
    """
    prompt = (
        "You will be given poems in Chinese, each identified by a unique checkpoint number. "
        "Evaluate these poems and rank them from best to worst, focusing on their alignment with the following qualities:\n"
        "• Modern free-verse style with introspective, existential themes\n"
        "• Vivid natural imagery\n"
        "• Exploration of life, identity, and the tension between belonging and escape\n"
        "• Philosophical or spiritual elements\n\n"
        "Consider these aspects in your critique:\n"
        "1. Tone - The poem's emotional ambience\n"
        "2. Style - The poet's writing approach or technique\n"
        "3. Imagery - Key visual elements that stand out\n"
        "4. Symbolism - Underlying symbolic meanings\n"
        "5. Themes - Central ideas explored\n"
        "6. Poetic Techniques - Use of figurative language such as metaphor\n"
        "7. Emotional Impact - How the poem's mood evolves\n"
        "8. Possible Expansions - Whether the poem provokes further reflection\n\n"
        "Return your evaluation as a JSON object with rankings and top poems."
    )
    # Add poems to the prompt for evaluation
    for i, poem in enumerate(poems, start=1):
        cp = poem.get("checkpoint_number", "N/A")
        body = poem.get("body", "").strip()
        prompt += f"{i}. [Checkpoint {cp}]\n{body}\n\n"
    return prompt
```

**Explanation:** The project implements a sophisticated two-stage evaluation pipeline. First, an NLP model (typically GPT-4) evaluates poems using detailed literary criteria including tone, imagery, symbolism, and emotional impact. This automated system efficiently filters hundreds of generated poems down to the most promising candidates. 

### Tournament Ranking Algorithm for Evaluation

```python
# From process3.py - Tournament-style evaluation for large poem sets 
def evaluate_batches_for_title(poems: List[dict], batch_size: int = 10, final_top: int = 5) -> List[dict]:
    """
    Partition poems into mini-batches, rank each batch, and select top candidates.
    This enables efficient processing of large numbers of poems in batch evaluation
    """
    batches = group_list(poems, batch_size)
    M = len(batches)
    if M == 0:
        return []
    # Calculate how many poems to select from each batch
    N = math.ceil(final_top / M)
    
    candidates = []
    for i, batch in enumerate(batches, start=1):
        print(f"Evaluating mini-batch {i}/{M} with {len(batch)} poems.")
        prompt = build_top_ranking_prompt(batch, desired=N)
        ranking_output = call_chatgpt_structured_top_fallback(prompt)
        
        if ranking_output:
            # Extract the top poems from this batch
            for cp in ranking_output.top_poems:
                for poem in batch:
                    if str(poem.get("checkpoint_number")) == str(cp):
                        candidates.append(poem)
                        break
        else:
            # Fallback in case of API failure
            candidates.append(batch[0])
    
    # Final ranking among all candidates
    return final_ranking(candidates, final_top=final_top)
```

**Explanation:** The implementation uses a tournament-style evaluation system to efficiently handle large numbers of generated poems. Instead of comparing all poems simultaneously (which would be computationally expensive and exceed context limits), the system divides poems into smaller batches, ranks each batch separately, and then performs a final evaluation on the winners from each group. This approach enables efficient processing of hundreds or thousands of poems while maintaining evaluation quality and consistency.



## 7. Human-in-the-Loop Curation

### Tkinter UI Implementation

```python
# From tk_app_accept_overall_poems.py - Human feedback evaluation interface
class PoemReviewerApp:
    def __init__(self, master, results):
        self.master = master
        self.master.title("Poem Reviewer")
        self.master.geometry("700x500")
        
        # Initialize data
        self.flattened_results = self.flatten_results(results)
        self.current_index = 0
        self.total = len(self.flattened_results)
        
        # Create UI components
        self.create_widgets()
        self.display_current_poem()
    
    def accept_poem(self):
        """Mark current poem as accepted for inclusion in training data"""
        poem = self.flattened_results[self.current_index]
        poem['accepted'] = True
        messagebox.showinfo("Accepted", f"Poem '{poem['image']}' has been accepted.")
        self.current_index += 1
        self.display_current_poem()
```

**Explanation:** The most promising poems generated and filtered from the automated system then go through a human-in-the-loop curation process via a custom Tkinter UI that enables careful selection, editing, and improvement. This hybrid approach combines algorithmic efficiency with human judgment to identify the most authentic and high-quality poems.



### Poem Editing Interface

```python
# From tk_app_accept_overall_poems.py - Detailed poem editing dialog
class EditDialog(tk.Toplevel):
    def __init__(self, master, poem, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Edit Poem")
        self.geometry("400x300")
        self.poem = poem
        self.result = None  # To store edited data
        
        # Make dialog modal
        self.transient(master)
        self.grab_set()
        
        # Variables to track what to edit
        self.edit_title_var = tk.BooleanVar()
        self.edit_body_var = tk.BooleanVar()
        
        # Create UI components
        self.checkbox_frame = ttk.Frame(self)
        self.checkbox_frame.pack(pady=10, padx=10, fill='x')
        
        # Add title and body editing options
        self.title_checkbox = ttk.Checkbutton(self.checkbox_frame, 
                                             text="Edit Title", 
                                             variable=self.edit_title_var, 
                                             command=self.toggle_fields)
        self.title_checkbox.pack(anchor='w')
        
        # ... more UI components ...
```

### Data Management

```python
# From tk_app_accept_overall_poems.py - Data transformation for UI
def flatten_results(self, results):
    """
    Convert nested list structure into flat list for UI processing
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

## 8. Iterative Refinement Process

### Data Augmentation Workflow

The complete process for the iterative improvement cycle involves:

```
# Code for the iterative refinement process

1. Initial training with ~100 original poems
   mlx_lm.lora -c yaml/lora2.yaml

2. Generate ~1000 candidate poems with varied themes using the trained model
   python generate2.py -m mlx-community/Qwen2.5-7B-Instruct-8bit \
     -a adapter_lora2 -p "$(cat input/words.txt)" -o generated_poems -e 300

3. LLM-based evaluation and ranking
   python process2.py -i generated_poems.json -o ranked_poems.json

4. Human curation review of top-ranked poems through developed app UI 
   python tk_app_accept_overall_poems.py

5. Convert accepted poems to training format and add to dataset after human evaluation
   python utils/convert_selections.py -i updated_overall_poems.json -o data/additional_poems.jsonl

6. Merge with original training data
   cat data/train.jsonl data/additional_poems.jsonl > data/train_enriched.jsonl

7. Re-train with enriched dataset for iterative and continuous improvement
   mlx_lm.lora -c yaml/lora4.yaml  # Updated config with enriched data
```

**Explanation:** The implementation features a complete iterative refinement pipeline that progressively improves the model's capabilities. Starting with initial training on ~100 poems, the system generates ~1000 new candidates that are automatically ranked for quality. The highest-ranked poems undergo expert human review, with the best selections incorporated back into the training data. This creates a virtuous cycle where the model learns from its own best outputs, continuously improving in quality and stylistic authenticity. The implementation avoids traditional data augmentation techniques, instead focusing on authentic expansion through human-verified generation.

## 9. Technical Experiments and Results

### Checkpoint Selection Algorithm

```python
# From test_all_checkpoints.py - Systematic checkpoint evaluation
def main():
    # ... argument parsing ...
    
    # Gather adapter checkpoint files
    adapter_files = []
    pattern = re.compile(r'(\d+)_adapters\.safetensors$')
    for fname in os.listdir(args.adapter_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            if epoch >= args.start_epoch:
                adapter_files.append((epoch, os.path.join(adapter_dir, fname)))
    
    # Sort checkpoint files by epoch
    adapter_files.sort(key=lambda x: x[0])
    
    # Test each checkpoint with the same prompt
    with open(args.output, "w") as outf:
        for epoch, adapter_file in adapter_files:
            # Create temporary directory with adapter
            with tempfile.TemporaryDirectory() as temp_adapter_dir:
                temp_adapter_path = os.path.join(temp_adapter_dir, "adapters.safetensors")
                shutil.copy(adapter_file, temp_adapter_path)
                
                # Run generation with this checkpoint
                cmd = [
                    "mlx_lm.generate",
                    "--model", args.model,
                    "--adapter-path", temp_adapter_dir,
                    "--prompt", args.prompt
                ]
                # ... run command and capture output ...
                
                # Write results for comparison
                outf.write(f"==== Epoch {epoch} (Adapter file: {adapter_file}) ====\n")
                outf.write(generation + "\n\n")
```

### Metadata Encoding System

```python
# From process3.py - Efficient metadata encoding for tracking generation variants
def encode_checkpoint(base_checkpoint: int, template_index: int) -> int:
    """
    Encode template variant information into the checkpoint number for efficient tracking.
    
    Examples:
    - Template 1: returns base_checkpoint (e.g., 1200)
    - Template 2: returns base_checkpoint + 1 (e.g., 1201)
    - Template 3: returns base_checkpoint + 2 (e.g., 1202)
    - Template 4: returns base_checkpoint + 3 (e.g., 1203)
    - Template 5: returns base_checkpoint + 4 (e.g., 1204)
    """
    if template_index < 1 or template_index > 5:
        raise ValueError("template_index must be between 1 and 5")
    return base_checkpoint + (template_index - 1)

def decode_checkpoint(checkpoint: int) -> Tuple[int, int]:
    """
    Decode an encoded checkpoint number back to base checkpoint and template index.
    """
    base = (checkpoint // 100) * 100
    template_index = checkpoint - base + 1
    if template_index < 1 or template_index > 5:
        raise ValueError("Invalid template index (must be between 1 and 5).")
    return base, template_index
```

**Explanation:** This implementation uses an efficient metadata encoding system that embeds prompt template information directly into checkpoint numbers. This allows tracking of which specific formulation generated each poem without requiring additional storage or complex data structures. For example, checkpoint 1203 represents the base checkpoint 1200 with template variant 4. This encoding enables precise analysis of which prompt formulations work best for different themes, facilitating systematic improvement of the prompt engineering strategy.



## 10. Word Selection for Poem Generation

The project implements a diverse set of carefully chosen thematic prompts:

```python
# From input/words.txt - Selected themes prompts for generation (snippet)
河 干涸 虫翅 雪峰 雨 伞裂 碎雨 隐线 湖 莲痂 刺骨 轮痕 冬至 积雪 惧融 征服 寒潮 死蕾 汁絮 冻绿 
风巢 空巢 虚美 树尖 死亡 残阳 火舌 灰祭 呼吸机 续命 噬心 锈蚀 忍冬 心坠 白洞 深渊 蝶刃 蝶散 器离 溃刃 
暴雨 纸溃 透城 雨锥 猫山 叶脉 寄生 疏离 人厌 糖幕 流放 嫉蜜 冻蜂 叶冢 无痕 冻寂 痛苦 气泡 悬疑 囚笼 
自由 褪色 蝶空 蒸发 呼吸 雾形 虚骸 气痂 雪崩 隙裂 雪屑 执消 癌夜 噬暗 焰湮 自愈 颅树 血树 湮灭 疯星
```

**Explanation:** The prompt selection strategy uses carefully curated thematic words that probe the model's capacity for poetic expression. These include concrete natural imagery ("河"/river, "雪峰"/snow peak), abstract emotional concepts ("疏离"/alienation, "悬疑"/suspense), and compound words with metaphorical potential ("莲痂"/lotus scab, "叶冢"/leaf mound). This approach tests the model's ability to handle both familiar concrete concepts and more abstract or unusual juxtapositions, ensuring it can generate poetry across the full spectrum of complexity present in the original corpus.

These words are carefully selected to probe the model's capacity for:

1. Concrete natural imagery (河/river, 雪峰/snow peak)
2. Abstract emotional concepts (疏离/alienation, 悬疑/suspense)
3. Compound words with metaphorical potential (莲痂/lotus scab, 叶冢/leaf mound)
4. Unusual juxtapositions (蝶刃/butterfly blade, 血树/blood tree)

## Conclusion

This project demonstrates a sophisticated, end-to-end approach to personalizing large language models for creative tasks. By combining LoRA fine-tuning, MLX framework efficiency, and a careful evaluation pipeline, it achieves technical innovation in three key areas:

1. **Parameter-Efficient Style Transfer**: Capturing unique poetic voice with minimal training data
2. **Consumer Hardware Accessibility**: Enabling training of 7B parameter models on personal devices
3. **Human-AI Collaborative Creation**: Establishing an iterative refinement process that preserves human artistic direction

The technical implementation bridges computational efficiency with artistic sensitivity, creating a framework for personalized creative AI that can be extended to other domains and languages.
