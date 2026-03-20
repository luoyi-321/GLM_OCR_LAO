# GLM-OCR Training Plan
> **Model:** [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) — 0.9B parameters
> **Stack:** LLaMA-Factory · LoRA / Full Fine-Tune · ShareGPT Format
> **Last updated:** March 2026

---

## Table of Contents
1. [Model Overview](#1-model-overview)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Dataset Registration](#4-dataset-registration)
5. [Training Configuration (YAML)](#5-training-configuration-yaml)
6. [Run Training](#6-run-training)
7. [Evaluation](#7-evaluation)
8. [Inference & Deployment](#8-inference--deployment)
9. [Hardware Requirements](#9-hardware-requirements)
10. [Troubleshooting Tips](#10-troubleshooting-tips)

---

## 1. Model Overview

GLM-OCR is a compact **0.9B parameter** multimodal OCR model from [Z.ai](https://docs.z.ai/guides/vlm/glm-ocr).

| Component | Details |
|---|---|
| Visual Encoder | CogViT (0.4B) — pre-trained on large image-text data |
| Connector | Lightweight cross-modal connector with token downsampling |
| Language Decoder | GLM-0.5B |
| Total Parameters | **~0.9B** |
| Benchmark Score | **94.62** on OmniDocBench V1.5 (Rank #1) |
| Throughput | 1.86 pages/sec (PDF) · 0.67 images/sec |

**Supported Tasks:**
- ✅ Text recognition (printed & handwritten)
- ✅ Formula recognition (LaTeX)
- ✅ Table recognition (Markdown / HTML)
- ✅ Information extraction from documents
- ✅ Multi-language documents

**Training Innovation:** Uses **Multi-Token Prediction (MTP)** — the decoder predicts multiple future tokens in parallel, improving training efficiency and stabilizing long-sequence generation.

---

## 2. Environment Setup

### Step 1 — Install GLM-OCR SDK

```bash
# Option A: Cloud/MaaS + local images/PDFs (lightest install)
pip install glmocr

# Option B: Self-hosted with layout detection (recommended for training)
pip install "glmocr[selfhosted]"

# Option C: Flask server support
pip install "glmocr[server]"

# Option D: Install from source
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr
pip install -e .
```

### Step 2 — Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Step 3 — Download Model Weights

```bash
# Via Hugging Face CLI
pip install huggingface_hub
huggingface-cli download zai-org/GLM-OCR --local-dir ./models/GLM-OCR

# OR via ModelScope (China mirror)
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('zai-org/GLM-OCR', local_dir='./models/GLM-OCR')"
```

> **Note:** You can also skip this step and set `model_name_or_path: zai-org/GLM-OCR` in the YAML config — weights will download automatically at training start.

---

## 3. Data Preparation

### ShareGPT Format (Required by LLaMA-Factory)

Each training sample must follow this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nPlease extract all text from this document."
    },
    {
      "role": "assistant",
      "content": "The document contains the following text:\n\n[extracted text here]"
    }
  ],
  "images": [
    "path/to/your/image.png"
  ]
}
```

**⚠️ Critical Rules:**
- The number of `<image>` tags in `content` must **exactly match** the number of entries in the `images` list.
- Image paths are **relative to the `data/` directory** of the LLaMA-Factory repository.
- Place all images inside `LLaMA-Factory/data/` and reference them relatively.

### Build Your Dataset File

Create a `my_ocr_dataset.json` file containing a list of samples:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>\nExtract the text from this invoice."
      },
      {
        "role": "assistant",
        "content": "Invoice #1234\nDate: 2026-03-01\nAmount: $500.00"
      }
    ],
    "images": ["ocr_data/invoice_001.png"]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "<image>\nConvert the table in this image to Markdown."
      },
      {
        "role": "assistant",
        "content": "| Name | Age | City |\n|------|-----|------|\n| Alice | 30 | NYC |"
      }
    ],
    "images": ["ocr_data/table_001.png"]
  }
]
```

### Recommended Dataset Size

| Use Case | Min Samples | Recommended |
|---|---|---|
| Domain adaptation (e.g., invoices) | 200–500 | 1,000+ |
| New language support | 1,000+ | 5,000+ |
| General improvement | 5,000+ | 10,000+ |

---

## 4. Dataset Registration

Edit `LLaMA-Factory/data/dataset_info.json` and add your dataset entry:

```json
{
  "my_ocr_dataset": {
    "file_name": "my_ocr_dataset.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

Then place your dataset file at:
```
LLaMA-Factory/data/my_ocr_dataset.json
LLaMA-Factory/data/ocr_data/        ← your images go here
```

---

## 5. Training Configuration (YAML)

### LoRA Fine-Tune (Recommended — ~8GB VRAM)

Create `glm_ocr_lora.yaml`:

```yaml
### Model
model_name_or_path: zai-org/GLM-OCR   # or local path: ./models/GLM-OCR

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### Dataset
dataset: my_ocr_dataset
template: glm_ocr
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 4

### Output
output_dir: ./outputs/glm_ocr_lora
logging_steps: 10
save_steps: 500
save_total_limit: 3
overwrite_output_dir: true

### Training
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
```

### Full Fine-Tune (~24GB VRAM)

Create `glm_ocr_full.yaml`:

```yaml
### Model
model_name_or_path: zai-org/GLM-OCR

### Method
stage: sft
do_train: true
finetuning_type: full

### Dataset
dataset: my_ocr_dataset
template: glm_ocr
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true

### Output
output_dir: ./outputs/glm_ocr_full
logging_steps: 10
save_steps: 500
save_total_limit: 3
overwrite_output_dir: true

### Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
optim: adamw_torch

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

---

## 6. Run Training

### Single GPU

```bash
cd LLaMA-Factory

# LoRA fine-tune
llamafactory-cli train glm_ocr_lora.yaml

# Full fine-tune
llamafactory-cli train glm_ocr_full.yaml
```

### Multi-GPU (Distributed Training)

```bash
# 2 GPUs with DeepSpeed ZeRO-2
FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
  llamafactory-cli train glm_ocr_lora.yaml \
  --deepspeed examples/deepspeed/ds_z2_config.json
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir ./outputs/glm_ocr_lora/runs
```

### Expected Output Structure After Training

```
outputs/
└── glm_ocr_lora/
    ├── adapter_config.json        ← LoRA adapter config
    ├── adapter_model.safetensors  ← LoRA weights
    ├── tokenizer_config.json
    └── runs/                      ← TensorBoard logs
```

---

## 7. Evaluation

### Merge LoRA Weights (for export)

```bash
llamafactory-cli export \
  model_name_or_path=zai-org/GLM-OCR \
  adapter_name_or_path=./outputs/glm_ocr_lora \
  template=glm_ocr \
  finetuning_type=lora \
  export_dir=./models/GLM-OCR-finetuned \
  export_size=4 \
  export_legacy_format=false
```

### Run Inference Test

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch

model_path = "./models/GLM-OCR-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

image = Image.open("test_doc.png").convert("RGB")
query = tokenizer.from_list_format([
    {"image": "test_doc.png"},
    {"text": "Please extract all text from this document."}
])

response, _ = model.chat(tokenizer, query=query, history=[])
print(response)
```

### Benchmark Against OmniDocBench

```bash
# Using the official glmocr SDK for evaluation
glmocr eval --model ./models/GLM-OCR-finetuned --benchmark omnidocbench
```

---

## 8. Inference & Deployment

### Option A — GLM-OCR SDK (Easiest)

```bash
export GLMOCR_API_KEY=your_api_key

# CLI usage
glmocr parse document.pdf

# Python usage
from glmocr import GLMOCRClient
client = GLMOCRClient()
result = client.parse("document.pdf")
print(result.text)
```

### Option B — vLLM (High-Throughput Server)

```bash
# Install latest transformers first
pip install git+https://github.com/huggingface/transformers

# Serve
vllm serve zai-org/GLM-OCR --port 8000
```

### Option C — SGLang

```bash
python -m sglang.launch_server \
  --model zai-org/GLM-OCR \
  --port 30000
```

### Option D — Ollama (Local, No GPU Required)

```bash
ollama run glm-ocr
```

---

## 9. Hardware Requirements

| Mode | GPU VRAM | GPU Type (Example) | Speed |
|---|---|---|---|
| LoRA Fine-Tune | **~8 GB** | RTX 3080/4070, T4 | Slow |
| Full Fine-Tune | **~24 GB** | RTX 3090/4090, A100 | Fast |
| Inference (bf16) | **~4 GB** | RTX 3060, V100 16G | 1.86 p/s |
| Inference (Ollama) | CPU only | Any machine | Slower |

**Recommended Training Setup:**
- 1× NVIDIA A100 40GB or 80GB (best)
- 1× NVIDIA RTX 4090 24GB (LoRA only)
- 2× NVIDIA RTX 3090 (LoRA with DeepSpeed)

---

## 10. Troubleshooting Tips

| Problem | Solution |
|---|---|
| `<image>` count mismatch | Ensure each `<image>` tag has a matching entry in `images` list |
| CUDA OOM during training | Reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps` |
| Model weights not found | Run `huggingface-cli download zai-org/GLM-OCR` first |
| Slow convergence | Increase `lora_rank` to 32 or 64; lower learning rate to `5e-5` |
| Template error | Confirm `template: glm_ocr` in your YAML config |
| Multi-GPU hangs | Add `ddp_timeout: 180000000` to YAML |

---

## Quick Reference — Full Pipeline

```
1. pip install "glmocr[selfhosted]"
2. git clone LLaMA-Factory → pip install
3. Prepare images + ShareGPT JSON dataset
4. Register dataset in dataset_info.json
5. Write training YAML (LoRA or Full)
6. llamafactory-cli train your_config.yaml
7. Export merged model
8. Deploy via SDK / vLLM / Ollama
```

---

## Useful Links

- [GLM-OCR GitHub](https://github.com/zai-org/GLM-OCR)
- [Fine-tuning Guide (LLaMA-Factory)](https://github.com/zai-org/GLM-OCR/blob/main/examples/finetune/README.md)
- [GLM-OCR Technical Report (arXiv:2603.10910)](https://arxiv.org/abs/2603.10910)
- [HF Transformers Docs](https://huggingface.co/docs/transformers/model_doc/glm_ocr)
- [vLLM GLM-OCR Recipe](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM-OCR.html)
- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [Z.ai Developer Docs](https://docs.z.ai/guides/vlm/glm-ocr)
- [Ollama GLM-OCR](https://ollama.com/library/glm-ocr)
