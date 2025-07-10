#!/usr/bin/env python3
"""
ZamAI Model Checker and Fixer
- Check all models on HF Hub for proper model files
- Download and fix models that are missing files
- Create proper project structure with spaces and datasets folders
"""

import os
import sys
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, login, Repository, snapshot_download, upload_file, create_repo
import shutil

def load_credentials():
    """Load credentials from HF-Credentials.txt"""
    try:
        with open("HF-Credentials.txt", "r") as f:
            lines = f.readlines()
        username = lines[0].split(":")[1].strip().strip("<>")
        token = lines[1].split(":")[1].strip().strip("<>")
        return username, token
    except Exception as e:
        print(f"❌ Failed to load credentials: {e}")
        sys.exit(1)

def list_user_models(username, token):
    """List all models from the user"""
    api = HfApi()
    try:
        models = api.list_models(author=username, token=token)
        return list(models)
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []

def check_model_files(model_id, token):
    """Check if a model repo has proper model files"""
    api = HfApi()
    try:
        # Get all files in the repo
        files = api.list_repo_files(repo_id=model_id, repo_type="model", token=token)
        
        # Check for common model files
        model_files = [f for f in files if f in ["pytorch_model.bin", "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]]
        
        # Check for model artifacts
        has_model_weights = any(f.endswith(".bin") or f.endswith(".safetensors") for f in files)
        has_config = "config.json" in files
        has_tokenizer_files = any("tokenizer" in f.lower() or f in ["vocab.json", "merges.txt"] for f in files)
        
        status = {
            "has_model_weights": has_model_weights,
            "has_config": has_config,
            "has_tokenizer_files": has_tokenizer_files,
            "model_files": model_files,
            "all_files": files,
            "complete": has_model_weights and has_config and has_tokenizer_files
        }
        
        return status
    except Exception as e:
        print(f"❌ Error checking model {model_id}: {e}")
        return {
            "has_model_weights": False,
            "has_config": False,
            "has_tokenizer_files": False,
            "model_files": [],
            "all_files": [],
            "complete": False,
            "error": str(e)
        }

def create_project_structure():
    """Create proper project structure with spaces and datasets folders"""
    # Create spaces directory
    spaces_dir = os.path.join(os.getcwd(), "spaces")
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    
    if not os.path.exists(spaces_dir):
        os.makedirs(spaces_dir)
        print(f"✅ Created spaces directory: {spaces_dir}")
    else:
        print(f"ℹ️ Spaces directory already exists: {spaces_dir}")
    
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
        print(f"✅ Created datasets directory: {datasets_dir}")
    else:
        print(f"ℹ️ Datasets directory already exists: {datasets_dir}")
    
    # Create README for spaces
    spaces_readme = os.path.join(spaces_dir, "README.md")
    if not os.path.exists(spaces_readme):
        with open(spaces_readme, "w") as f:
            f.write("""# ZamAI Spaces

This directory contains code and configuration for Hugging Face Spaces:

- Each subdirectory represents a Space
- Contains Gradio apps for training, fine-tuning and testing models
- Uses ZeroGPU for efficient GPU acceleration
""")
        print(f"✅ Created spaces README")
    
    # Create README for datasets
    datasets_readme = os.path.join(datasets_dir, "README.md")
    if not os.path.exists(datasets_readme):
        with open(datasets_readme, "w") as f:
            f.write("""# ZamAI Datasets

This directory contains datasets for training and fine-tuning models:

- Data for different model types
- Pre-processed training examples
- Test datasets
""")
        print(f"✅ Created datasets README")
    
    return spaces_dir, datasets_dir

def download_model_locally(model_id, token, output_dir=None):
    """Download a model locally"""
    if output_dir is None:
        # Create directory structure: models/model_name
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join("models", model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"📥 Downloading model {model_id} to {output_dir}...")
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            token=token,
            local_dir=output_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.jpg", "*.png", "*.txt"]
        )
        print(f"✅ Model downloaded successfully to {output_dir}")
        return True, output_dir
    except Exception as e:
        print(f"❌ Failed to download model {model_id}: {e}")
        return False, str(e)

def download_all_model_files(models, username, token):
    """Download all model files from HF Hub to local models directory"""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    downloaded_models = {}
    
    for i, model in enumerate(models, 1):
        model_id = model.id
        model_name = model_id.split("/")[-1]
        output_dir = os.path.join(models_dir, model_name)
        
        print(f"[{i}/{len(models)}] Processing {model_id}...")
        
        # Check if the model already exists locally
        if os.path.exists(output_dir) and os.listdir(output_dir):
            print(f"ℹ️ Model already exists locally: {output_dir}")
            downloaded_models[model_id] = {
                "local_path": output_dir,
                "status": "already_exists"
            }
            continue
        
        # Download the model
        success, result = download_model_locally(model_id, token, output_dir)
        if success:
            downloaded_models[model_id] = {
                "local_path": output_dir,
                "status": "downloaded"
            }
        else:
            downloaded_models[model_id] = {
                "local_path": None,
                "status": "failed",
                "error": result
            }
    
    return downloaded_models

def create_model_summary(models, username, token):
    """Create a comprehensive summary of all models"""
    model_summary = {}
    
    for i, model in enumerate(models, 1):
        model_id = model.id
        model_name = model_id.split("/")[-1]
        print(f"[{i}/{len(models)}] Checking model {model_id}...")
        
        # Check model files
        file_status = check_model_files(model_id, token)
        
        model_summary[model_id] = {
            "name": model_name,
            "id": model_id,
            "last_modified": str(model.last_modified),
            "complete": file_status["complete"],
            "has_model_weights": file_status["has_model_weights"],
            "has_config": file_status["has_config"],
            "has_tokenizer_files": file_status["has_tokenizer_files"],
            "model_files": file_status["model_files"],
            "file_count": len(file_status["all_files"]),
            "space_name": f"{username}/{model_name}-space"
        }
    
    return model_summary

def create_sample_datasets():
    """Create sample datasets for model training and fine-tuning"""
    datasets_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Create sample datasets for different tasks
    datasets = {
        "translation": {
            "filename": "pashto_english_translation.json",
            "data": [
                {"pashto": "ستاسو نوم څه دی؟", "english": "What is your name?"},
                {"pashto": "زه د افغانستان يم", "english": "I am from Afghanistan"},
                {"pashto": "نن هوا ښه ده", "english": "The weather is good today"},
                {"pashto": "دا کتاب ډیر په زړه پوري دی", "english": "This book is very interesting"},
                {"pashto": "زه پښتو زده کوم", "english": "I am learning Pashto"}
            ]
        },
        "sentiment": {
            "filename": "pashto_sentiment.json",
            "data": [
                {"text": "دا فلم ډیر ښه و", "label": "positive"},
                {"text": "زه له دې خواړو څخه خوند نه اخلم", "label": "negative"},
                {"text": "نن هوا ډیره ښه ده", "label": "positive"},
                {"text": "دا کتاب ډیر په زړه پوري دی", "label": "positive"},
                {"text": "زه دا خوښ نه کوم", "label": "negative"}
            ]
        },
        "qa": {
            "filename": "pashto_qa.json",
            "data": [
                {"question": "د افغانستان پلازمېنه څه ده؟", "answer": "کابل"},
                {"question": "پښتو د کوم هېواد رسمي ژبه ده؟", "answer": "افغانستان او پاکستان"},
                {"question": "د پښتو الفبا څومره توري لري؟", "answer": "۴۴"},
                {"question": "پښتو په کومې ژبنۍ کورنۍ پورې اړه لري؟", "answer": "هندو-اروپايي"},
                {"question": "د افغانستان لوی ښارونه کوم دي؟", "answer": "کابل، هرات، مزار شريف، کندهار، جلال آباد"}
            ]
        },
        "chat": {
            "filename": "pashto_chat.json",
            "data": [
                {"user": "سلام، څنګه یې؟", "assistant": "سلام، زه ښه یم، تاسو څنګه یاست؟"},
                {"user": "نن هوا څنګه ده؟", "assistant": "نن هوا ډیره ښه ده، لمر ځلیږي."},
                {"user": "ستاسو نوم څه دی؟", "assistant": "زه یو ژبنی مډل یم چې د پښتو ژبې سره مرسته کوم."},
                {"user": "په افغانستان کې څو رسمي ژبې دي؟", "assistant": "په افغانستان کې دوه رسمي ژبې دي: پښتو او دري."},
                {"user": "ما ته د پښتو په اړه معلومات راکړئ", "assistant": "پښتو د هندو-اروپايي ژبو کورنۍ اړوند يوه ژبه ده، چې د افغانستان او پاکستان په ځينو سيمو کې ويل کېږي. پښتو په عمده توګه په دوو لهجو وېشل شوې: شمالي او سويلي."}
            ]
        }
    }
    
    for dataset_type, dataset_info in datasets.items():
        filepath = os.path.join(datasets_dir, dataset_info["filename"])
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset_info["data"], f, ensure_ascii=False, indent=2)
            print(f"✅ Created sample dataset: {filepath}")
    
    # Create a README with dataset descriptions
    datasets_readme = os.path.join(datasets_dir, "README.md")
    with open(datasets_readme, "w") as f:
        f.write("""# ZamAI Datasets

This directory contains sample datasets for training and fine-tuning ZamAI models:

1. **Translation Dataset** (`pashto_english_translation.json`)
   - Pashto to English translation pairs
   - Use with `zamai-translator-pashto-en` model

2. **Sentiment Analysis Dataset** (`pashto_sentiment.json`)
   - Pashto text with sentiment labels (positive/negative)
   - Use with `zamai-sentiment-pashto` model

3. **Question Answering Dataset** (`pashto_qa.json`)
   - Pashto questions with their answers
   - Use with `zamai-qa-pashto` model

4. **Chat Dataset** (`pashto_chat.json`)
   - Pashto conversation examples (user/assistant)
   - Use with `zamai-pashto-chat-8b` and other chat models

## How to Use

These datasets can be loaded in the respective model Spaces:

```python
import json

# Load a dataset
with open("path_to_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Use for training/fine-tuning
```
""")
    print(f"✅ Created detailed datasets README")
    
    return datasets

def create_space_templates(model_summary, username, token):
    """Create template files for Spaces in the spaces directory"""
    spaces_dir = os.path.join(os.getcwd(), "spaces")
    os.makedirs(spaces_dir, exist_ok=True)
    
    for model_id, model_info in model_summary.items():
        model_name = model_info["name"]
        space_dir = os.path.join(spaces_dir, f"{model_name}-space")
        
        # Skip if directory already exists
        if os.path.exists(space_dir) and os.listdir(space_dir):
            print(f"ℹ️ Space directory already exists: {space_dir}")
            continue
        
        os.makedirs(space_dir, exist_ok=True)
        
        # Create app.py
        app_py = os.path.join(space_dir, "app.py")
        with open(app_py, "w") as f:
            f.write(f"""import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_NAME = "{model_id}"

@spaces.GPU
def load_model():
    \"\"\"Load the model and tokenizer\"\"\"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        return None, None

@spaces.GPU
def test_model(input_text):
    \"\"\"Test the model with given input\"\"\"
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "Failed to load model"
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {{str(e)}}"

# Create Gradio interface
with gr.Blocks(title="{model_name} Space") as iface:
    gr.Markdown(f"# {model_name}")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input", lines=3)
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            output_text = gr.Textbox(label="Output", lines=3)
    
    submit_btn.click(fn=test_model, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    iface.launch()
""")
        
        # Create requirements.txt
        req_txt = os.path.join(space_dir, "requirements.txt")
        with open(req_txt, "w") as f:
            f.write("""gradio>=4.0.0
spaces
torch
transformers>=4.30.0
""")
        
        # Create README.md
        readme_md = os.path.join(space_dir, "README.md")
        with open(readme_md, "w") as f:
            f.write(f"""---
title: {model_name} Space
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu-a10g
---

# {model_name} Space

This is a Space for the {model_name} model. You can:

1. Test the model
2. Train the model
3. Fine-tune the model

Uses ZeroGPU for efficient GPU acceleration.
""")
        
        print(f"✅ Created Space template for {model_name}")
    
    return spaces_dir

def main():
    # Load credentials
    username, token = load_credentials()
    
    try:
        login(token=token)
        print(f"✅ Successfully authenticated as {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    # Create project structure
    print("\n📂 Setting up project structure...")
    spaces_dir, datasets_dir = create_project_structure()
    
    # List models
    print("\n🔍 Fetching your models from Hugging Face Hub...")
    models = list_user_models(username, token)
    if not models:
        print("❌ No models found on your Hugging Face account.")
        sys.exit(1)
    
    print(f"📦 Found {len(models)} models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model.id}")
    
    # Create model summary
    print("\n🔍 Creating model summary...")
    model_summary = create_model_summary(models, username, token)
    
    # Save model summary
    with open("model_summary.json", "w") as f:
        json.dump(model_summary, f, indent=2)
    print(f"✅ Saved model summary to model_summary.json")
    
    # Count complete vs incomplete models
    complete_models = sum(1 for model in model_summary.values() if model["complete"])
    incomplete_models = len(model_summary) - complete_models
    
    print(f"\n📊 Model Status Summary:")
    print(f"  ✅ Complete models: {complete_models}")
    print(f"  ❌ Incomplete models: {incomplete_models}")
    
    # Download all model files
    print("\n📥 Downloading all model files...")
    downloaded_models = download_all_model_files(models, username, token)
    
    # Create sample datasets
    print("\n📊 Creating sample datasets...")
    create_sample_datasets()
    
    # Create Space templates
    print("\n🚀 Creating Space templates...")
    create_space_templates(model_summary, username, token)
    
    print("\n🎉 Project setup complete!")
    print(f"✅ Created project structure with:")
    print(f"  - Models directory with {len(downloaded_models)} models")
    print(f"  - Spaces directory with templates")
    print(f"  - Datasets directory with sample datasets")
    print("\nNext steps:")
    print("1. Run 'python model_manager.py' to manage your models")
    print("2. Run 'python create_remaining_spaces.py' to create Spaces on HF Hub")
    print("3. Explore and use the datasets in the 'datasets' directory")

if __name__ == "__main__":
    main()
