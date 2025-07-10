#!/usr/bin/env python3
"""
Script to manage Hugging Face Spaces - delete existing and create new ones
"""

import os
import sys
from huggingface_hub import HfApi, login, create_repo, delete_repo
import argparse
import time

def setup_hf_auth():
    """Setup Hugging Face authentication"""
    token = input("Enter your Hugging Face token: ").strip()
    if not token:
        print("❌ No token provided. Exiting.")
        sys.exit(1)
    
    try:
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face")
        return token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

def list_user_spaces(username, token):
    """List all spaces for a user"""
    api = HfApi()
    try:
        spaces = api.list_repos(author=username, repo_type="space", token=token)
        return [repo.id for repo in spaces]
    except Exception as e:
        print(f"❌ Failed to list spaces: {e}")
        return []

def delete_space(space_id, token):
    """Delete a Hugging Face Space"""
    api = HfApi()
    try:
        delete_repo(repo_id=space_id, repo_type="space", token=token)
        print(f"✅ Deleted space: {space_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete space {space_id}: {e}")
        return False

def create_model_space(model_name, username, token):
    """Create a new space for a model with train/finetune/test functionality"""
    space_name = f"{username}/{model_name}-space"
    
    try:
        # Create the space repository
        create_repo(
            repo_id=space_name,
            repo_type="space",
            token=token,
            private=False,
            exist_ok=True
        )
        
        print(f"✅ Created space: {space_name}")
        return space_name
        
    except Exception as e:
        print(f"❌ Failed to create space {space_name}: {e}")
        return None

def generate_space_files(model_name, username):
    """Generate the files needed for the Hugging Face Space"""
    
    # README.md for the space
    readme_content = f"""---
title: {model_name} Training Space
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# {model_name} Training Space

This space provides three main functionalities for the {model_name} model:

1. **Train**: Train the model from scratch
2. **Fine-tune**: Fine-tune the existing model
3. **Test**: Test the model with sample inputs

The space uses ZeroGPU for efficient GPU computation.
"""

    # app.py for the Gradio interface
    app_content = f'''import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import os

# Model configuration
MODEL_NAME = "{username}/{model_name}"

@spaces.GPU
def load_model():
    """Load the model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        return model, tokenizer
    except Exception as e:
        return None, None

@spaces.GPU
def test_model(input_text, max_length=100, temperature=0.7):
    """Test the model with given input"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "❌ Failed to load model. Please check if the model exists on Hugging Face Hub."
    
    try:
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(input_text):].strip()
        
    except Exception as e:
        return f"❌ Error during generation: {{str(e)}}"

def train_model(dataset_text, epochs=1, learning_rate=2e-5):
    """Train the model (placeholder implementation)"""
    return f"🚀 Training started with {{epochs}} epochs and learning rate {{learning_rate}}\\n\\nNote: This is a placeholder. Actual training requires more setup and computational resources."

def finetune_model(dataset_text, epochs=1, learning_rate=5e-5):
    """Fine-tune the model (placeholder implementation)"""
    return f"🔧 Fine-tuning started with {{epochs}} epochs and learning rate {{learning_rate}}\\n\\nNote: This is a placeholder. Actual fine-tuning requires more setup and computational resources."

# Create Gradio interface
with gr.Blocks(title="{model_name} Training Space", theme=gr.themes.Soft()) as iface:
    gr.Markdown(f"# {model_name} Training Space")
    gr.Markdown("Choose your operation: Train, Fine-tune, or Test the model")
    
    with gr.Tabs():
        # Test Tab
        with gr.TabItem("🧪 Test Model"):
            gr.Markdown("### Test the model with your input")
            with gr.Row():
                with gr.Column():
                    test_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to test the model...",
                        lines=3
                    )
                    max_length_slider = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        label="Max Length"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        label="Temperature"
                    )
                    test_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column():
                    test_output = gr.Textbox(
                        label="Model Output",
                        lines=5,
                        interactive=False
                    )
            
            test_btn.click(
                fn=test_model,
                inputs=[test_input, max_length_slider, temperature_slider],
                outputs=test_output
            )
        
        # Train Tab
        with gr.TabItem("🏋️ Train Model"):
            gr.Markdown("### Train the model from scratch")
            train_dataset = gr.Textbox(
                label="Training Dataset",
                placeholder="Upload or paste your training data...",
                lines=5
            )
            with gr.Row():
                train_epochs = gr.Number(label="Epochs", value=1, minimum=1)
                train_lr = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-6)
            
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Output", lines=5, interactive=False)
            
            train_btn.click(
                fn=train_model,
                inputs=[train_dataset, train_epochs, train_lr],
                outputs=train_output
            )
        
        # Fine-tune Tab
        with gr.TabItem("🔧 Fine-tune Model"):
            gr.Markdown("### Fine-tune the existing model")
            finetune_dataset = gr.Textbox(
                label="Fine-tuning Dataset",
                placeholder="Upload or paste your fine-tuning data...",
                lines=5
            )
            with gr.Row():
                finetune_epochs = gr.Number(label="Epochs", value=1, minimum=1)
                finetune_lr = gr.Number(label="Learning Rate", value=5e-5, minimum=1e-6)
            
            finetune_btn = gr.Button("🔧 Start Fine-tuning", variant="primary")
            finetune_output = gr.Textbox(label="Fine-tuning Output", lines=5, interactive=False)
            
            finetune_btn.click(
                fn=finetune_model,
                inputs=[finetune_dataset, finetune_epochs, finetune_lr],
                outputs=finetune_output
            )

if __name__ == "__main__":
    iface.launch()
'''

    # requirements.txt for the space
    requirements_content = """gradio==4.36.1
spaces
torch
transformers
datasets
accelerate
"""

    return readme_content, app_content, requirements_content

def main():
    parser = argparse.ArgumentParser(description="Manage Hugging Face Spaces")
    parser.add_argument("--username", help="Hugging Face username")
    parser.add_argument("--delete-all", action="store_true", help="Delete all existing spaces")
    parser.add_argument("--create-spaces", action="store_true", help="Create new spaces for models")
    parser.add_argument("--models", nargs="*", help="List of model names to create spaces for")
    
    args = parser.parse_args()
    
    # Setup authentication
    token = setup_hf_auth()
    
    # Get username
    if not args.username:
        username = input("Enter your Hugging Face username: ").strip()
    else:
        username = args.username
    
    if not username:
        print("❌ No username provided. Exiting.")
        sys.exit(1)
    
    api = HfApi()
    
    # Delete existing spaces if requested
    if args.delete_all:
        print("🔍 Fetching existing spaces...")
        spaces = list_user_spaces(username, token)
        
        if not spaces:
            print("ℹ️  No spaces found to delete")
        else:
            print(f"📦 Found {len(spaces)} spaces:")
            for space in spaces:
                print(f"  - {space}")
            
            confirm = input(f"\\nAre you sure you want to delete all {len(spaces)} spaces? (y/N): ").strip().lower()
            if confirm == 'y':
                deleted_count = 0
                for space in spaces:
                    if delete_space(space, token):
                        deleted_count += 1
                    time.sleep(1)  # Rate limiting
                
                print(f"\\n✅ Deleted {deleted_count}/{len(spaces)} spaces")
            else:
                print("❌ Deletion cancelled")
    
    # Create new spaces if requested
    if args.create_spaces:
        if not args.models:
            # Get model names from the models directory
            models_dir = "./models"
            if os.path.exists(models_dir):
                model_names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
            else:
                model_names = []
            
            if not model_names:
                print("❌ No models found. Please specify model names with --models")
                sys.exit(1)
        else:
            model_names = args.models
        
        print(f"🚀 Creating spaces for {len(model_names)} models...")
        
        created_count = 0
        for model_name in model_names:
            print(f"\\n📦 Creating space for {model_name}...")
            space_name = create_model_space(model_name, username, token)
            
            if space_name:
                # Generate and save space files locally for manual upload
                readme, app, requirements = generate_space_files(model_name, username)
                
                space_dir = f"./spaces/{model_name}-space"
                os.makedirs(space_dir, exist_ok=True)
                
                with open(f"{space_dir}/README.md", "w") as f:
                    f.write(readme)
                
                with open(f"{space_dir}/app.py", "w") as f:
                    f.write(app)
                
                with open(f"{space_dir}/requirements.txt", "w") as f:
                    f.write(requirements)
                
                print(f"📁 Space files created in {space_dir}")
                print(f"🔗 You can now upload these files to: https://huggingface.co/spaces/{space_name}")
                
                created_count += 1
            
            time.sleep(2)  # Rate limiting
        
        print(f"\\n✅ Created {created_count}/{len(model_names)} spaces")

if __name__ == "__main__":
    main()
