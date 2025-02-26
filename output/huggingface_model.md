---
base_model: mlx-community/Qwen2.5-7B-Instruct-8bit
language:
- en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE
pipeline_tag: text-generation
tags:
- chat
- mlx
- mlx
---

# jerryzhao173985/Qwen2.5-7B-poem-8bit

The Model [jerryzhao173985/Qwen2.5-7B-poem-8bit](https://huggingface.co/jerryzhao173985/Qwen2.5-7B-poem-8bit) was
converted to MLX format from [mlx-community/Qwen2.5-7B-Instruct-8bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-8bit)
using mlx-lm version **0.21.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("jerryzhao173985/Qwen2.5-7B-poem-8bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
