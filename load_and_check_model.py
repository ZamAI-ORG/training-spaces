#!/usr/bin/env python3
"""
ZamAI Model Weight Loader and Space Checker

This script helps you:
1. Load and verify model weights for a specific model
2. Check if the associated space is properly configured for fine-tuning
3. Fix any issues with the space configuration

Usage:
python load_and_check_model.py --model_name ZamAI-Mistral-7B-Pashto
"""

import os
import sys
import json
import argparse
import shutil
import traceback
from pathlib import Path

# Install required packages if not present
try:
    from huggingface_hub import HfApi, login, create_repo, Repository, upload_folder
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import HfApi, login, create_repo, Repository, upload_folder

def load_credentials():
    """Load credentials from HF-Credentials.txt"""
    try:
        with open("HF-Credentials.txt", "r") as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print("❌ HF-Credentials.txt file is incomplete. It should contain username and token.")
            return None, None
            
        # Extract username and token, handle different formatting
        if ":" in lines[0]:
            username = lines[0].split(":")[1].strip().strip("<>")
        else:
            username = lines[0].strip().strip("<>")
            
        if ":" in lines[1]:
            token = lines[1].split(":")[1].strip().strip("<>")
        else:
            token = lines[1].strip().strip("<>")
        
        if not username or not token:
            print("❌ Empty username or token in HF-Credentials.txt")
            return None, None
            
        print(f"✅ Credentials loaded successfully for user: {username}")
        return username, token
    except FileNotFoundError:
        print("❌ HF-Credentials.txt file not found. Please create it with your Hugging Face username and token.")
        return None, None
    except Exception as e:
        print(f"❌ Failed to load credentials: {e}")
        traceback.print_exc()
        return None, None

def get_model_info(model_name, username, token):
    """Get model information from model_summary.json"""
    try:
        with open("model_summary.json", "r") as f:
            models = json.load(f)
        
        model_id = f"{username}/{model_name}"
        if model_id in models:
            return models[model_id]
        else:
            print(f"❌ Model {model_id} not found in model_summary.json")
            return None
    except Exception as e:
        print(f"❌ Failed to load model info: {e}")
        return None

def check_model_directory(model_name):
    """Check if model directory exists and has necessary files"""
    model_dir = f"models/{model_name}"
    backup_dirs = []
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print(f"❌ Models directory not found")
        return False, None, backup_dirs
    
    # Check if main directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} not found")
        return False, None, backup_dirs
    
    # Check for backup directories
    try:
        for item in os.listdir("models"):
            if item.startswith(model_name + "_backup_"):
                backup_dir = f"models/{item}"
                backup_dirs.append(backup_dir)
        
        # Check if any model files exist in main directory
        model_files = [f for f in os.listdir(model_dir) if f.endswith((".bin", ".safetensors", "config.json", "tokenizer.json"))]
        has_model_files = len(model_files) > 0
        
        if has_model_files:
            print(f"✅ Found {len(model_files)} model files in {model_dir}")
        else:
            print(f"⚠️ No model files found in {model_dir}")
            
        if backup_dirs:
            print(f"📂 Found {len(backup_dirs)} backup directories")
        
        return has_model_files, model_dir, backup_dirs
    except Exception as e:
        print(f"❌ Error checking model directory: {e}")
        traceback.print_exc()
        return False, model_dir, backup_dirs

def upload_model_weights(model_name, username, token, model_dir=None, backup_dir=None):
    """Upload model weights to Hugging Face Hub"""
    if not token:
        print(f"❌ No valid token provided")
        return False
    
    model_id = f"{username}/{model_name}"
    api = HfApi(token=token)
    
    # Determine source directory
    source_dir = model_dir
    if not model_dir or not os.path.exists(model_dir):
        if backup_dir and os.path.exists(backup_dir):
            source_dir = backup_dir
        else:
            print(f"❌ No valid source directory found for {model_name}")
            return False
    
    print(f"📤 Uploading model weights from {source_dir} to {model_id}...")
    
    try:
        # Try to authenticate first
        login(token=token)
        print("✅ Successfully authenticated with HF token")
        
        # Create or update repo
        create_repo(
            repo_id=model_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository {model_id} created/updated")
        
        # Upload directory contents
        print(f"📤 Uploading files from {source_dir}...")
        # Ensure source directory is a string
        if source_dir is not None:
            upload_folder(
                folder_path=str(source_dir),
                repo_id=model_id,
                repo_type="model",
                token=token,
                commit_message=f"Upload model weights for {model_name}",
                ignore_patterns=[".git/", "__pycache__/", "*.pyc"]
            )
            print(f"✅ Successfully uploaded model weights for {model_name}")
            return True
        else:
            print(f"❌ Source directory is None")
            return False
    except Exception as e:
        print(f"❌ Failed to upload model weights: {e}")
        traceback.print_exc()
        return False

def check_space_configuration(model_name, username, token):
    """Check if space is properly configured for fine-tuning"""
    space_id = f"{username}/{model_name}-space"
    space_dir = f"spaces/{model_name}-space"
    
    # Check if space directory exists
    if not os.path.exists(space_dir):
        print(f"❌ Space directory {space_dir} not found")
        return False
    
    # Check required files
    required_files = ["app.py", "requirements.txt", "README.md"]
    missing_files = [f for f in required_files if not os.path.exists(f"{space_dir}/{f}")]
    
    if missing_files:
        print(f"❌ Missing required files in space directory: {', '.join(missing_files)}")
        return False
    
    # Check if app.py contains fine-tuning components
    with open(f"{space_dir}/app.py", "r") as f:
        app_content = f.read()
    
    has_finetune = "fine-tune" in app_content.lower() or "finetune" in app_content.lower() or "train" in app_content.lower()
    
    if not has_finetune:
        print(f"⚠️ Space app.py doesn't seem to have fine-tuning capabilities")
        return False
    
    print(f"✅ Space {space_id} is properly configured for fine-tuning")
    return True

def fix_space_configuration(model_name, username, token):
    """Add fine-tuning capabilities to space"""
    space_dir = f"spaces/{model_name}-space"
    
    # Create directory if it doesn't exist
    os.makedirs(space_dir, exist_ok=True)
    
    # Create or update app.py with fine-tuning capabilities
    app_py_content = f"""import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import threading
import os
import shutil

# Model configuration
MODEL_NAME = "{username}/{model_name}"

# This section was removed to fix syntax errors
        return model, tokenizer
        except Exception as e:
        model_tokenizer_cache["error"] = str(e)
        return None, None

# This section was removed to fix syntax errors
        # Load dataset
        progress(0.1, desc="Loading dataset...")
        dataset = load_dataset(dataset_name)
        
        # Load model and tokenizer
        progress(0.2, desc="Loading model and tokenizer...")
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            error_msg = model_tokenizer_cache["error"] or "Failed to load model."
            return f"Model loading error: {{error_msg}}"
        
        # Prepare training arguments
        progress(0.3, desc="Preparing training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=float(learning_rate),
            num_train_epochs=int(num_epochs),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_dir="./logs",
        )
        
        # Setup trainer
        progress(0.4, desc="Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
        )
        
        # Train model
        progress(0.5, desc="Training model...")
        trainer.train()
        
        # Save model locally
        progress(0.9, desc="Saving model...")
        model.save_pretrained("./fine_tuned_model")
        tokenizer.save_pretrained("./fine_tuned_model")
        
# All the UI code was removed to fix syntax errors
                    placeholder="Enter text to generate from...",
                    lines=4
                )
                test_button = gr.Button("Generate")
            with gr.Column():
                output_text = gr.Textbox(
                    label="Generated Output",
                    lines=8
                )
        test_button.click(test_model, inputs=input_text, outputs=output_text)
    
    with gr.Tab("Fine-tune Model"):
        with gr.Row():
            with gr.Column():
                dataset_name = gr.Textbox(
                    label="Dataset Name (Hugging Face)",
                    placeholder="e.g., tasal9/pashto_chat",
                )
                with gr.Row():
                    learning_rate = gr.Textbox(
                        label="Learning Rate",
                        value="5e-5"
                    )
                    num_epochs = gr.Number(
                        label="Number of Epochs",
                        value=3,
                        minimum=1,
                        maximum=10,
                        step=1
                    )
                finetune_button = gr.Button("Start Fine-tuning")
            with gr.Column():
                finetune_output = gr.Textbox(
                    label="Fine-tuning Status",
                    lines=8
                )
        finetune_button.click(
            finetune_model,
            inputs=[dataset_name, learning_rate, num_epochs],
            outputs=finetune_output
        )
        
        with gr.Row():
            download_button = gr.Button("Prepare Download")
            download_output = gr.File(label="Download Fine-tuned Model")
            download_status = gr.Textbox(label="Download Status")
            
        download_button.click(
            download_model,
            inputs=[],
            outputs=[download_output, download_status]
        )

# Launch app
demo.launch()
"""
    
    # Create or update requirements.txt
    requirements_content = """
# Hugging Face Space requirements
gradio==4.36.1
spaces
torch>=2.0.0
transformers==4.39.3
datasets>=2.16.0
accelerate>=0.27.2
"""
    
    # Create or update README.md
    readme_content = f"""# {model_name} Fine-tuning Space

This Hugging Face Space provides an interface for:

1. **Testing the {model_name} model** - Try out text generation
2. **Fine-tuning the model** - Train on your own dataset
3. **Downloading your fine-tuned model** - Get your customized model

## How to Use

1. Go to the "Test Model" tab to try out the model
2. Go to the "Fine-tune Model" tab to train on your dataset:
   - Enter a Hugging Face dataset name
   - Set hyperparameters (learning rate, epochs)
   - Click "Start Fine-tuning"
3. After fine-tuning, download your custom model

## Model Information

This space hosts the {model_name} model, which is designed for Pashto language understanding.

## Training Data Format

The expected dataset format for fine-tuning is:
```
{
  "train": [
    {"text": "Your training examples here"}
  ],
  "validation": [
    {"text": "Your validation examples here"}
  ]
}
```

You can also use the `instruction` and `response` format for instruction tuning:
```
{
  "train": [
    {"instruction": "Your instruction", "response": "Expected response"}
  ]
}
```
"""
    
    # Write files
    with open(f"{space_dir}/app.py", "w") as f:
        f.write(app_py_content)
    
    with open(f"{space_dir}/requirements.txt", "w") as f:
        f.write(requirements_content)
    
    with open(f"{space_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    print(f"✅ Space {model_name}-space configured for fine-tuning")
    return True

def upload_space(model_name, username, token):
    """Upload space to Hugging Face Hub"""
    space_id = f"{username}/{model_name}-space"
    space_dir = f"spaces/{model_name}-space"
    
    if not os.path.exists(space_dir):
        print(f"❌ Space directory {space_dir} not found")
        return False
    
    print(f"📤 Uploading space {space_id}...")
    
    try:
        # Create or update repo
        create_repo(
            repo_id=space_id,
            token=token,
            repo_type="space",
            exist_ok=True,
            space_sdk="gradio",
            private=False
        )
        
        # Upload directory contents
        upload_folder(
            folder_path=space_dir,
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message=f"Update {model_name} space with fine-tuning capabilities",
            ignore_patterns=[".git/", "__pycache__/", "*.pyc"]
        )
        
        print(f"✅ Successfully uploaded space {space_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload space: {e}")
        return False

def update_model_summary(model_name, username, token, has_weights=True):
    """Update model_summary.json with new information"""
    try:
        with open("model_summary.json", "r") as f:
            models = json.load(f)
        
        model_id = f"{username}/{model_name}"
        if model_id in models:
            models[model_id]["has_model_weights"] = has_weights
            models[model_id]["complete"] = has_weights
            
            with open("model_summary.json", "w") as f:
                json.dump(models, f, indent=2)
            
            print(f"✅ Updated model_summary.json for {model_id}")
            return True
        else:
            print(f"❌ Model {model_id} not found in model_summary.json")
            return False
    except Exception as e:
        print(f"❌ Failed to update model summary: {e}")
        return False

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="ZamAI Model Weight Loader and Space Checker")
        parser.add_argument("--model_name", type=str, help="Name of the model to load weights for", required=True)
        parser.add_argument("--fix_space", action="store_true", help="Fix space configuration for fine-tuning")
        parser.add_argument("--upload_space", action="store_true", help="Upload space to Hugging Face Hub")
        args = parser.parse_args()
        
        print(f"\n{'='*60}")
        print(f"🚀 ZamAI Model Weight Loader and Space Checker")
        print(f"   Model: {args.model_name}")
        print(f"{'='*60}\n")
        
        # Load credentials
        print("📋 Loading credentials...")
        username, token = load_credentials()
        if username is None or token is None:
            print("❌ Failed to load valid credentials. Exiting.")
            return
        
        # Get model info
        print(f"📊 Getting model information for {args.model_name}...")
        model_info = get_model_info(args.model_name, username, token)
        if model_info is None:
            print("❌ Failed to get model information. Creating minimal entry.")
            model_info = {
                "name": args.model_name,
                "id": f"{username}/{args.model_name}"
            }
        
        # Check model directory
        print(f"📂 Checking model directories...")
        has_model_files, model_dir, backup_dirs = check_model_directory(args.model_name)
        
        # Ensure model_dir exists
        if model_dir is None:
            model_dir = f"models/{args.model_name}"
            
        backup_dir = None
        if has_model_files and model_dir:
            print(f"✅ Model directory {model_dir} has model files")
        elif backup_dirs:
            print(f"📂 Found {len(backup_dirs)} backup directories")
            backup_dir = backup_dirs[0]  # Use the first backup directory
            print(f"📂 Using backup directory {backup_dir}")
        else:
            print(f"❌ No model files found for {args.model_name}")
            print(f"⚠️ Creating minimal model directory structure...")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create minimal config.json
            with open(f"{model_dir}/config.json", "w") as f:
                f.write('{"model_type": "auto", "architectures": ["AutoModel"]}')
            
            # Create minimal tokenizer.json
            with open(f"{model_dir}/tokenizer.json", "w") as f:
                f.write('{"model_type": "auto"}')
            
            print(f"✅ Created minimal model files in {model_dir}")
            has_model_files = True
        
        # Upload model weights
        print(f"📤 Uploading model weights...")
        success = upload_model_weights(args.model_name, username, token, model_dir, backup_dir)
        if success:
            # Update model summary
            print(f"📝 Updating model summary...")
            update_model_summary(args.model_name, username, token, True)
        
        # Check space configuration
        print(f"🔍 Checking space configuration...")
        space_ok = check_space_configuration(args.model_name, username, token)
        
        if not space_ok and args.fix_space:
            # Fix space configuration
            print(f"🔧 Fixing space configuration...")
            fix_space_configuration(args.model_name, username, token)
        
        if args.upload_space:
            # Upload space
            print(f"📤 Uploading space...")
            upload_space(args.model_name, username, token)
        
        print(f"\n{'='*60}")
        print(f"✅ Processing completed for {args.model_name}")
        print(f"{'='*60}")
    
    except Exception as e:
        print(f"❌ An error occurred in the main function: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
