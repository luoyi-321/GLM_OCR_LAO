#!/usr/bin/env python3
"""
GLM-OCR Training Pipeline
=========================
All-in-one script for setup, data preparation, training, and inference.

Usage:
    python glm_ocr_pipeline.py setup              # Install dependencies & download model
    python glm_ocr_pipeline.py prepare            # Prepare training data
    python glm_ocr_pipeline.py train              # Run training
    python glm_ocr_pipeline.py infer              # Run inference on images
    python glm_ocr_pipeline.py export             # Export LoRA weights to merged model
    python glm_ocr_pipeline.py serve              # Start inference server
    python glm_ocr_pipeline.py all                # Run full pipeline

Author: GLM-OCR Training Pipeline
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Paths
    "project_dir": Path(__file__).parent.absolute(),
    "llama_factory_dir": Path(__file__).parent / "LLaMA-Factory",
    "model_dir": Path(__file__).parent / "models" / "GLM-OCR",
    "output_dir": Path(__file__).parent / "outputs",
    "data_dir": Path(__file__).parent / "data",

    # Model
    "model_name": "zai-org/GLM-OCR",

    # Training defaults
    "training_mode": "lora",  # "lora" or "full"
    "num_epochs": 3,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,

    # Dataset
    "dataset_name": "my_ocr_dataset",
}


# =============================================================================
# Utility Functions
# =============================================================================

def run_command(cmd: list[str] | str, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"[CMD] {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n[Step {step}] {description}")
    print("-" * 40)


def get_llamafactory_base_command() -> list[str]:
    """Resolve LLaMA-Factory command in a cross-platform way."""
    cli_path = shutil.which("llamafactory-cli") or shutil.which("llamafactory-cli.exe")
    if cli_path:
        return [cli_path]

    # Fallback to module execution using the current Python interpreter.
    return [sys.executable, "-m", "llamafactory.cli"]


# =============================================================================
# Setup Functions
# =============================================================================

def setup_environment():
    """Install all dependencies and download model weights."""
    print_header("GLM-OCR Environment Setup")

    # Step 1: Install GLM-OCR SDK
    print_step(1, "Installing GLM-OCR SDK")
    run_command([sys.executable, "-m", "pip", "install", "glmocr[selfhosted]"], check=False)

    # Step 2: Clone and install LLaMA-Factory
    print_step(2, "Setting up LLaMA-Factory")
    llama_factory_dir = CONFIG["llama_factory_dir"]

    if not llama_factory_dir.exists():
        print("Cloning LLaMA-Factory repository...")
        run_command([
            "git", "clone",
            "https://github.com/hiyouga/LLaMA-Factory.git",
            str(llama_factory_dir)
        ])
    else:
        print("LLaMA-Factory already exists, updating...")
        run_command(["git", "pull"], cwd=llama_factory_dir)

    print("Installing LLaMA-Factory...")
    run_command([
        sys.executable, "-m", "pip", "install", "-e", ".[torch,metrics]"
    ], cwd=llama_factory_dir, check=False)

    # Step 3: Download model weights
    print_step(3, "Downloading GLM-OCR model weights")
    model_dir = CONFIG["model_dir"]

    if not model_dir.exists() or not any(model_dir.iterdir()):
        print("Downloading from Hugging Face...")
        run_command([
            sys.executable, "-m", "pip", "install", "huggingface_hub"
        ], check=False)
        run_command([
            "hf", "download",
            CONFIG["model_name"],
            "--local-dir", str(model_dir)
        ], check=False)
    else:
        print(f"Model already exists at {model_dir}")

    # Step 4: Create directory structure
    print_step(4, "Creating project directories")
    ensure_dir(CONFIG["data_dir"])
    ensure_dir(CONFIG["data_dir"] / "images")
    ensure_dir(CONFIG["output_dir"])
    ensure_dir(CONFIG["project_dir"] / "configs")

    # Step 5: Install additional dependencies
    print_step(5, "Installing additional dependencies")
    deps = ["torch", "transformers", "pillow", "tensorboard", "tqdm"]
    for dep in deps:
        run_command([sys.executable, "-m", "pip", "install", dep], check=False)

    print_header("Setup Complete!")
    print(f"""
Project structure created:
    {CONFIG['project_dir']}/
    ├── LLaMA-Factory/     (training framework)
    ├── models/GLM-OCR/    (model weights)
    ├── data/              (your training data)
    │   └── images/        (document images)
    ├── outputs/           (training outputs)
    └── configs/           (YAML configs)

Next steps:
    1. Add your images to: {CONFIG['data_dir']}/images/
    2. Run: python glm_ocr_pipeline.py prepare
    """)


# =============================================================================
# Data Preparation Functions
# =============================================================================

def create_sample_dataset():
    """Create a sample dataset structure for reference."""
    print_header("Data Preparation")

    data_dir = CONFIG["data_dir"]
    images_dir = data_dir / "images"
    ensure_dir(images_dir)

    # Create sample dataset JSON
    sample_data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nExtract all text from this document."
                },
                {
                    "role": "assistant",
                    "content": "# Sample Document\n\nThis is the extracted text from the document.\n\n- Item 1\n- Item 2\n- Item 3"
                }
            ],
            "images": ["images/sample_001.png"]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>\nConvert this table to Markdown format."
                },
                {
                    "role": "assistant",
                    "content": "| Column A | Column B | Column C |\n|----------|----------|----------|\n| Value 1  | Value 2  | Value 3  |"
                }
            ],
            "images": ["images/sample_002.png"]
        }
    ]

    dataset_path = data_dir / f"{CONFIG['dataset_name']}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample dataset created: {dataset_path}")
    print("""
Dataset Format (ShareGPT):
--------------------------
Each sample requires:
  - "messages": Array with user/assistant conversation
  - "images": Array of image paths (relative to data/ directory)

Important:
  - Number of <image> tags MUST match number of images
  - Place images in: data/images/
  - Reference as: images/filename.png
    """)

    return dataset_path


def _resolve_image_for_gt(gt_file: Path) -> Optional[Path]:
    """Resolve image path for a .gt.txt file."""
    base_path = Path(str(gt_file)[:-7])  # remove ".gt.txt"
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def convert_splits_to_sharegpt(splits_dir: Path):
    """Convert split directories (train/val/test) with .png + .gt.txt pairs to ShareGPT JSON."""
    data_dir = CONFIG["data_dir"]
    images_dir = ensure_dir(data_dir / "images")
    dataset_name = CONFIG["dataset_name"]

    split_entries: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    missing_images = 0

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / split_name
        if not split_path.exists():
            print(f"  [WARN] Split directory not found: {split_path}")
            continue

        dst_split_images = ensure_dir(images_dir / split_name)
        gt_files = sorted(split_path.rglob("*.gt.txt"))
        print(f"  Processing {split_name}: found {len(gt_files)} labels")

        for gt_file in gt_files:
            image_file = _resolve_image_for_gt(gt_file)
            if image_file is None:
                missing_images += 1
                continue

            text = gt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            dst_image = dst_split_images / image_file.name
            shutil.copy2(image_file, dst_image)

            split_entries[split_name].append({
                "messages": [
                    {"role": "user", "content": "<image>\nExtract all text from this document."},
                    {"role": "assistant", "content": text}
                ],
                "images": [f"images/{split_name}/{dst_image.name}"]
            })

        split_dataset_path = data_dir / f"{dataset_name}_{split_name}.json"
        with open(split_dataset_path, "w", encoding="utf-8") as f:
            json.dump(split_entries[split_name], f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(split_entries[split_name])} samples -> {split_dataset_path.name}")

    main_dataset = split_entries["train"]
    if not main_dataset:
        main_dataset = split_entries["train"] + split_entries["val"] + split_entries["test"]

    dataset_path = data_dir / f"{dataset_name}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(main_dataset, f, indent=2, ensure_ascii=False)

    print(f"Main training dataset saved: {dataset_path} ({len(main_dataset)} samples)")
    if missing_images > 0:
        print(f"[WARN] Skipped {missing_images} labels without matching images")


def prepare_data(
    source_dir: Optional[str] = None,
    annotations_file: Optional[str] = None,
    splits_dir: Optional[str] = None,
):
    """Prepare training data from source images and annotations."""
    print_header("Preparing Training Data")

    data_dir = CONFIG["data_dir"]
    images_dir = data_dir / "images"

    # Preferred path: convert pre-split OCR data
    if splits_dir:
        splits_path = Path(splits_dir)
        if not splits_path.exists():
            print(f"ERROR: splits directory not found: {splits_dir}")
            sys.exit(1)

        print_step(1, f"Converting split data from {splits_dir}")
        convert_splits_to_sharegpt(splits_path)

    # Legacy path: source images + annotations
    elif source_dir:
        source_path = Path(source_dir)
        if source_path.exists():
            print_step(1, f"Copying images from {source_dir}")
            for img in source_path.glob("*"):
                if img.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
                    shutil.copy2(img, images_dir / img.name)
                    print(f"  Copied: {img.name}")

        # Check if we have annotations
        if annotations_file:
            annotations_path = Path(annotations_file)
            if annotations_path.exists():
                print_step(2, "Processing annotations file")
                convert_annotations_to_sharegpt(annotations_path)
        else:
            print_step(2, "Creating sample dataset structure")
            create_sample_dataset()

    else:
        # Auto-detect local splits/ if available
        default_splits = CONFIG["project_dir"] / "splits"
        if default_splits.exists():
            print_step(1, f"Converting split data from {default_splits}")
            convert_splits_to_sharegpt(default_splits)
        else:
            print_step(1, "Creating sample dataset structure")
            create_sample_dataset()

    # Register dataset in LLaMA-Factory
    print_step(3, "Registering dataset in LLaMA-Factory")
    register_dataset()

    print_header("Data Preparation Complete!")


def convert_annotations_to_sharegpt(annotations_path: Path):
    """Convert various annotation formats to ShareGPT format."""
    data_dir = CONFIG["data_dir"]

    # Try to load the annotations
    with open(annotations_path, "r", encoding="utf-8") as f:
        content = f.read()

    dataset = []

    # Try JSON format first
    try:
        annotations = json.loads(content)

        # Handle different JSON structures
        if isinstance(annotations, list):
            for item in annotations:
                sample = convert_annotation_item(item)
                if sample:
                    dataset.append(sample)
        elif isinstance(annotations, dict):
            for key, value in annotations.items():
                sample = convert_annotation_item(value, image_name=key)
                if sample:
                    dataset.append(sample)
    except json.JSONDecodeError:
        # Try line-by-line format (image_path \t text)
        for line in content.strip().split("\n"):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                image_path, text = parts[0], "\t".join(parts[1:])
                dataset.append({
                    "messages": [
                        {"role": "user", "content": "<image>\nExtract all text from this document."},
                        {"role": "assistant", "content": text}
                    ],
                    "images": [f"images/{Path(image_path).name}"]
                })

    # Save dataset
    dataset_path = data_dir / f"{CONFIG['dataset_name']}.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(dataset)} samples to: {dataset_path}")


def convert_annotation_item(item: dict, image_name: Optional[str] = None) -> Optional[dict]:
    """Convert a single annotation item to ShareGPT format."""
    # Extract image path
    image = image_name or item.get("image") or item.get("file") or item.get("filename")
    if not image:
        return None

    # Extract text/label
    text = item.get("text") or item.get("label") or item.get("content") or item.get("transcription")
    if not text:
        return None

    # Extract task type if available
    task = item.get("task") or item.get("type") or "ocr"

    # Build appropriate prompt based on task
    prompts = {
        "ocr": "Extract all text from this document.",
        "table": "Convert this table to Markdown format.",
        "formula": "Convert this formula to LaTeX.",
        "layout": "Describe the layout and extract text from this document.",
    }
    prompt = prompts.get(task.lower(), prompts["ocr"])

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": text}
        ],
        "images": [f"images/{Path(image).name}"]
    }


def register_dataset():
    """Register the dataset in LLaMA-Factory's dataset_info.json."""
    llama_factory_dir = CONFIG["llama_factory_dir"]
    data_info_path = llama_factory_dir / "data" / "dataset_info.json"

    # Ensure LLaMA-Factory data directory exists
    ensure_dir(llama_factory_dir / "data")

    # Load existing dataset info or create new
    if data_info_path.exists():
        with open(data_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    # Add our primary dataset
    dataset_name = CONFIG["dataset_name"]
    _sharegpt_entry = {
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }

    dataset_info[dataset_name] = {"file_name": f"{dataset_name}.json", **_sharegpt_entry}

    # Add split datasets if available
    for split_name in ["train", "val", "test"]:
        split_dataset = CONFIG["data_dir"] / f"{dataset_name}_{split_name}.json"
        if split_dataset.exists():
            split_key = f"{dataset_name}_{split_name}"
            dataset_info[split_key] = {"file_name": split_dataset.name, **_sharegpt_entry}

    # Save updated dataset info
    with open(data_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    # Copy our dataset and images to LLaMA-Factory data directory
    src_data_dir = CONFIG["data_dir"]
    dst_data_dir = llama_factory_dir / "data"

    # Copy dataset JSON
    src_dataset = src_data_dir / f"{dataset_name}.json"
    if src_dataset.exists():
        shutil.copy2(src_dataset, dst_data_dir / f"{dataset_name}.json")

    # Copy images directory
    src_images = src_data_dir / "images"
    dst_images = dst_data_dir / "images"
    if src_images.exists():
        if dst_images.exists():
            shutil.rmtree(dst_images)
        shutil.copytree(src_images, dst_images)

    print(f"Dataset '{dataset_name}' registered in LLaMA-Factory")


# =============================================================================
# Training Configuration Functions
# =============================================================================

def create_training_config(mode: str = "lora") -> Path:
    """Create a YAML training configuration file."""
    print_step(1, f"Creating {mode.upper()} training configuration")

    configs_dir = ensure_dir(CONFIG["project_dir"] / "configs")
    model_path = CONFIG["model_dir"] if CONFIG["model_dir"].exists() else CONFIG["model_name"]

    if mode == "lora":
        config = f"""### Model
model_name_or_path: {model_path}

### Method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {CONFIG['lora_rank']}
lora_alpha: {CONFIG['lora_alpha']}
lora_dropout: 0.05
lora_target: all

### Dataset
dataset: {CONFIG['dataset_name']}
template: glm_ocr
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 4

### Output
output_dir: {CONFIG['output_dir']}/glm_ocr_lora
logging_steps: 10
save_steps: 500
save_total_limit: 3
overwrite_output_dir: true

### Training
per_device_train_batch_size: {CONFIG['batch_size']}
gradient_accumulation_steps: 8
learning_rate: {CONFIG['learning_rate']}
num_train_epochs: {CONFIG['num_epochs']}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 500
"""
        config_path = configs_dir / "glm_ocr_lora.yaml"
    else:
        config = f"""### Model
model_name_or_path: {model_path}

### Method
stage: sft
do_train: true
finetuning_type: full

### Dataset
dataset: {CONFIG['dataset_name']}
template: glm_ocr
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true

### Output
output_dir: {CONFIG['output_dir']}/glm_ocr_full
logging_steps: 10
save_steps: 500
save_total_limit: 3
overwrite_output_dir: true

### Training
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-5
num_train_epochs: {CONFIG['num_epochs']}
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
optim: adamw_torch

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
"""
        config_path = configs_dir / "glm_ocr_full.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config)

    print(f"Config saved: {config_path}")
    return config_path


# =============================================================================
# Training Functions
# =============================================================================

def train(mode: str = "lora", config_path: Optional[Path] = None):
    """Run the training process."""
    print_header(f"GLM-OCR {mode.upper()} Training")

    llama_factory_dir = CONFIG["llama_factory_dir"]

    # Check if LLaMA-Factory is installed
    if not llama_factory_dir.exists():
        print("ERROR: LLaMA-Factory not found. Run 'python glm_ocr_pipeline.py setup' first.")
        sys.exit(1)

    llama_factory_cmd = get_llamafactory_base_command()

    # Create or use existing config
    if config_path is None:
        config_path = create_training_config(mode)

    # Copy config to LLaMA-Factory directory
    dst_config = llama_factory_dir / config_path.name
    shutil.copy2(config_path, dst_config)

    print_step(2, "Starting training")
    print(f"Config: {config_path}")
    print(f"Mode: {mode.upper()}")
    print(f"Output: {CONFIG['output_dir']}/glm_ocr_{mode}")
    print()

    # Run training
    try:
        run_command(llama_factory_cmd + ["train", str(dst_config.name)], cwd=llama_factory_dir)
    except FileNotFoundError:
        print("ERROR: LLaMA-Factory CLI not found in current environment.")
        print("Run: python glm_ocr_pipeline.py setup")
        print("Or install manually: python -m pip install -e .[torch,metrics] (inside LLaMA-Factory)")
        sys.exit(1)

    print_header("Training Complete!")
    print(f"""
Results saved to: {CONFIG['output_dir']}/glm_ocr_{mode}

Next steps:
    1. Check training logs: tensorboard --logdir {CONFIG['output_dir']}/glm_ocr_{mode}/runs
    2. Export model: python glm_ocr_pipeline.py export
    3. Run inference: python glm_ocr_pipeline.py infer --image path/to/image.png
    """)


# =============================================================================
# Export Functions
# =============================================================================

def export_model(adapter_path: Optional[str] = None):
    """Export LoRA weights to a merged standalone model."""
    print_header("Exporting Model")

    llama_factory_dir = CONFIG["llama_factory_dir"]

    if adapter_path is None:
        adapter_path = CONFIG["output_dir"] / "glm_ocr_lora"
    else:
        adapter_path = Path(adapter_path)

    export_dir = CONFIG["project_dir"] / "models" / "GLM-OCR-finetuned"

    print_step(1, "Merging LoRA weights")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {export_dir}")

    model_path = CONFIG["model_dir"] if CONFIG["model_dir"].exists() else CONFIG["model_name"]

    llama_factory_cmd = get_llamafactory_base_command()

    run_command(llama_factory_cmd + [
        "export",
        f"model_name_or_path={model_path}",
        f"adapter_name_or_path={adapter_path}",
        "template=glm_ocr",
        "finetuning_type=lora",
        f"export_dir={export_dir}",
        "export_size=4",
        "export_legacy_format=false"
    ], cwd=llama_factory_dir)

    print_header("Export Complete!")
    print(f"Merged model saved to: {export_dir}")


# =============================================================================
# Inference Functions
# =============================================================================

def infer(image_path: str, model_path: Optional[str] = None, prompt: Optional[str] = None):
    """Run inference on an image."""
    print_header("GLM-OCR Inference")

    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        print("ERROR: Required packages not installed. Run 'python glm_ocr_pipeline.py setup' first.")
        sys.exit(1)

    # Determine model path
    if model_path is None:
        finetuned = CONFIG["project_dir"] / "models" / "GLM-OCR-finetuned"
        if finetuned.exists():
            model_path = str(finetuned)
        elif CONFIG["model_dir"].exists():
            model_path = str(CONFIG["model_dir"])
        else:
            model_path = CONFIG["model_name"]

    print(f"Model: {model_path}")
    print(f"Image: {image_path}")

    if not Path(image_path).exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    # Load model
    print_step(1, "Loading model")
    model_path_obj = Path(model_path)
    adapter_config_path = model_path_obj / "adapter_config.json"

    if adapter_config_path.exists():
        try:
            PeftModel = importlib.import_module("peft").PeftModel
        except ImportError:
            print("ERROR: PEFT is required to load LoRA checkpoints for inference.")
            print("Install with: python -m pip install peft")
            sys.exit(1)

        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        base_model_path = adapter_config.get("base_model_name_or_path") or str(CONFIG["model_dir"])
        print(f"Adapter checkpoint detected: {model_path}")
        print(f"Base model: {base_model_path}")

        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, str(model_path_obj))
    else:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()

    # Build model inputs
    print_step(2, "Processing image")
    if prompt is None:
        prompt = "Text Recognition:"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    # Run inference
    print_step(3, "Running inference")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    response = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print_header("Result")
    print(response)

    return response


def batch_infer(input_dir: str, output_file: str = "results.json", model_path: Optional[str] = None):
    """Run inference on multiple images."""
    print_header("Batch Inference")

    input_path = Path(input_dir)
    results = []

    # Get all images
    images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))

    print(f"Found {len(images)} images")

    for i, img_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
        try:
            result = infer(str(img_path), model_path)
            results.append({
                "image": img_path.name,
                "text": result,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "image": img_path.name,
                "text": None,
                "status": "error",
                "error": str(e)
            })

    # Save results
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print_header("Batch Inference Complete!")
    print(f"Results saved to: {output_path}")


# =============================================================================
# Server Functions
# =============================================================================

def serve(port: int = 8000, model_path: Optional[str] = None):
    """Start an inference server using vLLM."""
    print_header("Starting Inference Server")

    if model_path is None:
        finetuned = CONFIG["project_dir"] / "models" / "GLM-OCR-finetuned"
        if finetuned.exists():
            model_path = str(finetuned)
        else:
            model_path = CONFIG["model_name"]

    print(f"Model: {model_path}")
    print(f"Port: {port}")
    print(f"URL: http://localhost:{port}")

    run_command([
        "vllm", "serve", model_path, "--port", str(port)
    ])


# =============================================================================
# Full Pipeline
# =============================================================================

def run_full_pipeline(source_dir: Optional[str] = None, mode: str = "lora"):
    """Run the complete pipeline from setup to inference."""
    print_header("GLM-OCR Full Pipeline")

    # Step 1: Setup
    setup_environment()

    # Step 2: Data preparation
    prepare_data(source_dir)

    # Step 3: Training
    train(mode)

    # Step 4: Export
    if mode == "lora":
        export_model()

    print_header("Full Pipeline Complete!")
    print("""
Your GLM-OCR model is now trained and ready!

To run inference:
    python glm_ocr_pipeline.py infer --image path/to/document.png

To start a server:
    python glm_ocr_pipeline.py serve

To run batch inference:
    python glm_ocr_pipeline.py infer --input-dir path/to/images/ --output results.json
    """)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GLM-OCR Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python glm_ocr_pipeline.py setup
    python glm_ocr_pipeline.py prepare --source ./my_images --annotations labels.json
    python glm_ocr_pipeline.py prepare --splits-dir ./splits
    python glm_ocr_pipeline.py train --mode lora
    python glm_ocr_pipeline.py infer --image document.png
    python glm_ocr_pipeline.py all --source ./my_images
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Install dependencies and download model")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare training data")
    prepare_parser.add_argument("--source", "-s", help="Source directory with images")
    prepare_parser.add_argument("--annotations", "-a", help="Annotations file (JSON or TSV)")
    prepare_parser.add_argument("--splits-dir", help="Directory containing train/val/test with .png + .gt.txt pairs")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("--mode", "-m", choices=["lora", "full"], default="lora",
                              help="Training mode: lora (~8GB VRAM) or full (~24GB VRAM)")
    train_parser.add_argument("--config", "-c", help="Path to custom YAML config")
    train_parser.add_argument("--epochs", "-e", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, help="Batch size")
    train_parser.add_argument("--lr", type=float, help="Learning rate")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export LoRA weights to merged model")
    export_parser.add_argument("--adapter", help="Path to LoRA adapter directory")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--image", "-i", help="Path to single image")
    infer_parser.add_argument("--input-dir", "-d", help="Directory with images for batch inference")
    infer_parser.add_argument("--output", "-o", default="results.json", help="Output file for batch results")
    infer_parser.add_argument("--model", "-m", help="Path to model (default: auto-detect)")
    infer_parser.add_argument("--prompt", "-p", help="Custom prompt for inference")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start inference server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port")
    serve_parser.add_argument("--model", "-m", help="Path to model")

    # All command
    all_parser = subparsers.add_parser("all", help="Run full pipeline")
    all_parser.add_argument("--source", "-s", help="Source directory with images")
    all_parser.add_argument("--mode", "-m", choices=["lora", "full"], default="lora",
                            help="Training mode")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Update config with CLI arguments
    if hasattr(args, "epochs") and args.epochs:
        CONFIG["num_epochs"] = args.epochs
    if hasattr(args, "batch_size") and args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if hasattr(args, "lr") and args.lr:
        CONFIG["learning_rate"] = args.lr

    # Execute command
    if args.command == "setup":
        setup_environment()

    elif args.command == "prepare":
        prepare_data(args.source, args.annotations, args.splits_dir)

    elif args.command == "train":
        config_path = Path(args.config) if args.config else None
        train(args.mode, config_path)

    elif args.command == "export":
        export_model(args.adapter)

    elif args.command == "infer":
        if args.input_dir:
            batch_infer(args.input_dir, args.output, args.model)
        elif args.image:
            infer(args.image, args.model, args.prompt)
        else:
            print("ERROR: Specify --image or --input-dir")
            sys.exit(1)

    elif args.command == "serve":
        serve(args.port, args.model)

    elif args.command == "all":
        run_full_pipeline(args.source, args.mode)


if __name__ == "__main__":
    main()
