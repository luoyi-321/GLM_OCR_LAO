# GLM-OCR Fine-tuning Pipeline — Lao OCR & ID Card Extraction

Fine-tune [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B multimodal) on Lao text recognition and structured ID card / passport extraction using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Inference](#inference)
7. [Export](#export)
8. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
GLM_OCR_LAO/
├── glm_ocr_pipeline.py      # All-in-one pipeline CLI
├── run_pipeline.sh          # Bash shortcut (Git Bash / WSL)
├── configs/
│   └── glm_ocr_lora.yaml    # Active training config (edit here)
├── data/                    # Lao OCR dataset (ShareGPT JSON)
│   ├── my_ocr_dataset.json
│   ├── my_ocr_dataset_train.json
│   ├── my_ocr_dataset_val.json
│   ├── my_ocr_dataset_test.json
│   └── images/
│       ├── train/
│       ├── val/
│       └── test/
├── data_id_card/            # ID card extraction dataset
│   ├── train_sharegpt.json  # Converted ShareGPT format (265 samples)
│   ├── val_sharegpt.json    # Validation set (29 samples)
│   └── dataset/             # Source JPEG images
├── splits/                  # Raw Lao OCR data (image + .gt.txt pairs)
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── GLM-OCR/             # Downloaded base model weights
├── outputs/
│   ├── glm_ocr_lora/        # Lao OCR fine-tune checkpoints
│   └── glm_ocr_lora_idcard/ # ID card fine-tune checkpoints
└── LLaMA-Factory/           # Training framework (cloned)
    └── data/
        └── dataset_info.json
```

---

## Requirements

- Python 3.10+  
- CUDA GPU (8 GB VRAM minimum for LoRA, 24 GB for full fine-tune)  
- Conda environment (e.g. `minimind_env`)

```bash
conda activate minimind_env
pip install "glmocr[selfhosted]" peft huggingface_hub
pip install -e "LLaMA-Factory[torch,metrics]"
```

---

## Setup

Downloads the base model and creates the project directories:

```bash
python glm_ocr_pipeline.py setup
```

This will:
1. Install GLM-OCR SDK (`glmocr[selfhosted]`)
2. Clone and install LLaMA-Factory
3. Download `zai-org/GLM-OCR` weights to `models/GLM-OCR/`
4. Create `data/`, `outputs/`, `configs/` directories

---

## Data Preparation

### Lao OCR dataset (from `splits/`)

```bash
python glm_ocr_pipeline.py prepare --splits-dir ./splits
```

Reads `splits/train|val|test/*.gt.txt` + matching images → produces:
- `data/my_ocr_dataset_train.json` (501,924 samples)
- `data/my_ocr_dataset_val.json`   (62,740 samples)
- `data/my_ocr_dataset_test.json`  (62,741 samples)
- Registers all datasets in `LLaMA-Factory/data/dataset_info.json`

### ID card dataset (`data_id_card/`)

Pre-converted ShareGPT JSON files are ready at:
- `data_id_card/train_sharegpt.json` — 265 samples
- `data_id_card/val_sharegpt.json`   — 29 samples

Already registered as `id_card_train` / `id_card_val` in `dataset_info.json`.

---

## Training

> **Always pass `--config` to avoid the YAML being regenerated.**

### Lao OCR — LoRA from scratch

```bash
python glm_ocr_pipeline.py train --mode lora --config configs/glm_ocr_lora.yaml
```

### ID card — resume from Lao OCR checkpoint

Edit `configs/glm_ocr_lora.yaml` to set:

```yaml
model_name_or_path: D:\Sulixay_file\GLM_OCR_LAO\models\GLM-OCR
adapter_name_or_path: D:\Sulixay_file\GLM_OCR_LAO\outputs\glm_ocr_lora\checkpoint-1782

dataset: id_card_train
eval_dataset: id_card_val

output_dir: D:\Sulixay_file\GLM_OCR_LAO\outputs\glm_ocr_lora_idcard
```

Then run:

```bash
python glm_ocr_pipeline.py train --mode lora --config configs/glm_ocr_lora.yaml
```

### Key config options

| Parameter | Default | Notes |
|---|---|---|
| `lora_rank` | 16 | Higher = more capacity |
| `learning_rate` | 1e-4 | Lower (5e-5) for checkpoint resume |
| `num_train_epochs` | 3–10 | Increase for small datasets |
| `per_device_train_batch_size` | 2 | Reduce if OOM |
| `gradient_accumulation_steps` | 8 | Effective batch = 2×8=16 |

Monitor training:

```bash
tensorboard --logdir outputs/glm_ocr_lora/runs
```

---

## Inference

### Single image — Lao OCR

```bash
python glm_ocr_pipeline.py infer \
  --task ocr \
  --model outputs/glm_ocr_lora/checkpoint-1782 \
  --image path/to/image.png
```

### Single image — ID card extraction

```bash
python glm_ocr_pipeline.py infer \
  --task id_card \
  --model outputs/glm_ocr_lora_idcard/checkpoint-XXX \
  --image path/to/id_card.jpg
```

Returns structured JSON with fields: `name`, `surname`, `date_of_birth`, `card_number`, `issued_date`, `expired_date`, `nationality`, etc.

### Batch inference

```bash
python glm_ocr_pipeline.py infer \
  --task id_card \
  --model outputs/glm_ocr_lora_idcard/checkpoint-XXX \
  --input-dir path/to/images/ \
  --output results.json
```

### Custom prompt

```bash
python glm_ocr_pipeline.py infer \
  --image path/to/image.png \
  --prompt "Table Recognition:"
```

Supported built-in task prompts:

| `--task` | Prompt sent to model |
|---|---|
| `ocr` | `Text Recognition:` |
| `id_card` | `Extract ID/Passport information from this image in JSON format.` |

---

## Export

Merge LoRA adapter weights into a standalone model:

```bash
python glm_ocr_pipeline.py export
# or with a specific adapter:
python glm_ocr_pipeline.py export --adapter outputs/glm_ocr_lora_idcard/checkpoint-XXX
```

Merged model saved to `models/GLM-OCR-finetuned/`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `KeyError: 'from'` | Dataset missing `tags` block in `dataset_info.json` — `role`/`content` keys must be declared |
| `TypeError: can only concatenate list to list` | System prompt is a list object — convert JSONL to ShareGPT JSON first |
| `FileNotFoundError: llamafactory-cli` | Run `python glm_ocr_pipeline.py setup` or use `--config` with the correct env activated |
| `AttributeError: TokenizersBackend has no attribute from_list_format` | Use `AutoProcessor` + `apply_chat_template` — fixed in current pipeline |
| YAML reset on `train` | Always pass `--config configs/glm_ocr_lora.yaml` explicitly |
| OOM during training | Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps` |
