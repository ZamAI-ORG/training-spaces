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
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo, Repository, upload_folder

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
    
    # Check if main directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} not found")
        return False, None, backup_dirs
    
    # Check for backup directories
    for item in os.listdir("models"):
        if item.startswith(model_name + "_backup_"):
            backup_dir = f"models/{item}"
            backup_dirs.append(backup_dir)
    
    # Check if any model files exist in main directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith((".bin", ".safetensors", "config.json", "tokenizer.json"))]
    has_model_files = len(model_files) > 0
    
    return has_model_files, model_dir, backup_dirs

def upload_model_weights(model_name, username, token, model_dir=None, backup_dir=None):
    """Upload model weights to Hugging Face Hub"""
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
        # Create or update repo
        create_repo(
            repo_id=model_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        
        # Upload directory contents
        upload_folder(
            folder_path=source_dir,
            repo_id=model_id,
            repo_type="model",
            token=token,
            commit_message=f"Upload model weights for {model_name}",
            ignore_patterns=[".git/", "__pycache__/", "*.pyc"]
        )
        
        print(f"✅ Successfully uploaded model weights for {model_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload model weights: {e}")
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

# Cache model and tokenizer
model_tokenizer_cache = {{"model": None, "tokenizer": None, "loaded": False, "error": None}}
model_lock = threading.Lock()

@spaces.GPU
def load_model():
    """Load the model and tokenizer, cache them"""
    with model_lock:
        if model_tokenizer_cache["loaded"]:
            return model_tokenizer_cache["model"], model_tokenizer_cache["tokenizer"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model_tokenizer_cache["model"] = model
            model_tokenizer_cache["tokenizer"] = tokenizer
            model_tokenizer_cache["loaded"] = True
            model_tokenizer_cache["error"] = None
            return model, tokenizer
        except Exception as e:
            model_tokenizer_cache["error"] = str(e)
            return None, None

@spaces.GPU
def test_model(input_text):
    """Test the model with given input"""
    # Input validation
    if not isinstance(input_text, str) or len(input_text.strip()) == 0:
        return "Please enter some text to generate."
    if len(input_text) > 512:
        return "Input too long (max 512 characters)."

    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        error_msg = model_tokenizer_cache["error"] or "Failed to load model."
        return f"Model loading error: {{error_msg}}"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Generation error: {{str(e)}}"

@spaces.GPU
def finetune_model(dataset_name, learning_rate, num_epochs, progress=gr.Progress()):
    """Fine-tune the model on a dataset"""
    try:
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
        
        progress(1.0, desc="Complete!")
        return "Model fine-tuning complete! You can now download the fine-tuned model."
    except Exception as e:
        return f"Fine-tuning error: {{str(e)}}"

def download_model():
    """Create a downloadable zip of the fine-tuned model"""
    try:
        if not os.path.exists("./fine_tuned_model"):
            return None, "No fine-tuned model available. Please fine-tune the model first."
        
        # Create zip file
        if os.path.exists("./fine_tuned_model.zip"):
            os.remove("./fine_tuned_model.zip")
        
        shutil.make_archive("fine_tuned_model", "zip", ".", "fine_tuned_model")
        
        return "./fine_tuned_model.zip", "Fine-tuned model ready for download."
    except Exception as e:
        return None, f"Error creating zip: {{str(e)}}"

# Create Gradio interface
with gr.Blocks(title="{model_name} Demo & Fine-tuning") as demo:
    gr.Markdown(f"# {model_name} Demo & Fine-tuning")
    
    with gr.Tab("Test Model"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
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
    # Parse arguments
    parser = argparse.ArgumentParser(description="ZamAI Model Weight Loader and Space Checker")
    parser.add_argument("--model_name", type=str, help="Name of the model to load weights for", required=True)
    parser.add_argument("--fix_space", action="store_true", help="Fix space configuration for fine-tuning")
    parser.add_argument("--upload_space", action="store_true", help="Upload space to Hugging Face Hub")
    args = parser.parse_args()
    
    # Load credentials
    username, token = load_credentials()
    
    # Get model info
    model_info = get_model_info(args.model_name, username, token)
    if model_info is None:
        return
    
    # Check model directory
    has_model_files, model_dir, backup_dirs = check_model_directory(args.model_name)
    
    if has_model_files:
        print(f"✅ Model directory {model_dir} has model files")
        backup_dir = None
    elif backup_dirs:
        print(f"📂 Found {len(backup_dirs)} backup directories")
        backup_dir = backup_dirs[0]  # Use the first backup directory
        print(f"📂 Using backup directory {backup_dir}")
    else:
        print(f"❌ No model files found for {args.model_name}")
        return
    
    # Upload model weights
    success = upload_model_weights(args.model_name, username, token, model_dir, backup_dir)
    if success:
        # Update model summary
        update_model_summary(args.model_name, username, token, True)
    
    # Check space configuration
    space_ok = check_space_configuration(args.model_name, username, token)
    
    if not space_ok and args.fix_space:
        # Fix space configuration
        fix_space_configuration(args.model_name, username, token)
    
    if args.upload_space:
        # Upload space
        upload_space(args.model_name, username, token)

if __name__ == "__main__":
    main()
