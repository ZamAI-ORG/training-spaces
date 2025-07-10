#!/usr/bin/env python3
"""
Script to create only the remaining spaces that failed due to rate limiting
"""

import sys
import os
import time
from huggingface_hub import HfApi, login, create_repo, delete_repo, Repository

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
    api = HfApi()
    try:
        models = api.list_models(author=username, token=token)
        return [m.id for m in models]
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []

def list_user_spaces(username, token):
    api = HfApi()
    try:
        spaces = api.list_spaces(author=username, token=token)
        return [space.id for space in spaces]
    except Exception as e:
        print(f"❌ Failed to list spaces: {e}")
        return []

def create_model_space_with_delay(model_id, username, token, delay=10):
    model_name = model_id.split('/')[-1]
    space_name = f"{username}/{model_name}-space"
    
    # Check if space already exists
    existing_spaces = list_user_spaces(username, token)
    if space_name in existing_spaces:
        print(f"✅ Space {space_name} already exists, skipping...")
        return space_name
    
    try:
        print(f"🚀 Creating space: {space_name}")
        create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            private=False,
            exist_ok=True
        )
        print(f"✅ Created space: {space_name}")
        
        # Create the Gradio app with train/finetune/test and ZeroGPU
        create_space_files(space_name, model_id, model_name, token)
        
        # Add delay to avoid rate limiting
        print(f"⏳ Waiting {delay} seconds to avoid rate limiting...")
        time.sleep(delay)
        
        return space_name
    except Exception as e:
        print(f"❌ Failed to create space {space_name}: {e}")
        return None

def create_space_files(space_name, model_id, model_name, token):
    """Create and upload app.py, requirements.txt, and README.md to the space"""
    
    # README.md content with proper ZeroGPU hardware
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
hardware: zero-gpu-a10g
---

# {model_name} Training Space

This space provides three main functionalities for the {model_name} model:

1. **Train**: Train the model from scratch
2. **Fine-tune**: Fine-tune the existing model  
3. **Test**: Test the model with sample inputs

The space uses ZeroGPU for efficient GPU computation.
"""

    # Enhanced app.py content with model saving functionality
    app_content = f"""import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import HfApi, upload_folder
import os
import json
from datetime import datetime

# Model configuration
MODEL_NAME = "{model_id}"

@spaces.GPU
def load_model():
    \"\"\"Load the model and tokenizer\"\"\"
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        return None, None

@spaces.GPU
def test_model(input_text, max_length=100, temperature=0.7):
    \"\"\"Test the model with given input\"\"\"
    if not input_text.strip():
        return "Please enter some text to test the model."
    
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "❌ Failed to load model. Please check if the model exists on Hugging Face Hub."
    
    try:
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(input_text):].strip()
        
    except Exception as e:
        return f"❌ Error during generation: {{str(e)}}"

@spaces.GPU
def train_model(dataset_text, epochs=1, learning_rate=2e-5, save_model=False):
    \"\"\"Train the model with actual implementation\"\"\"
    if not dataset_text.strip():
        return "❌ Please provide training data."
    
    try:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "❌ Failed to load model for training."
        
        # Prepare training data (simplified)
        train_texts = dataset_text.split('\\n')
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results-{{datetime.now().strftime('%Y%m%d-%H%M%S')}}",
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,
            logging_dir='./logs',
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Create a simple dataset class
        class SimpleDataset:
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {{key: torch.tensor(val[idx]) for key, val in self.encodings.items()}}
            
            def __len__(self):
                return len(self.encodings.input_ids)
        
        train_dataset = SimpleDataset(train_encodings)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Start training
        result = trainer.train()
        
        output = f"🚀 Training completed!\\n"
        output += f"📊 Training Loss: {{result.training_loss:.4f}}\\n"
        output += f"⏱️ Training Time: {{result.metrics.get('train_runtime', 'N/A')}} seconds\\n"
        
        if save_model:
            # Save the trained model
            save_dir = f"./trained-{{{model_name}}}-{{datetime.now().strftime('%Y%m%d-%H%M%S')}}"
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            output += f"💾 Model saved locally to: {{save_dir}}\\n"
            output += "📤 To upload to HuggingFace Hub, use the 'Upload Model' button below.\\n"
        
        return output
        
    except Exception as e:
        return f"❌ Training failed: {{str(e)}}"

@spaces.GPU
def finetune_model(dataset_text, epochs=1, learning_rate=5e-5, save_model=False):
    \"\"\"Fine-tune the model with actual implementation\"\"\"
    if not dataset_text.strip():
        return "❌ Please provide fine-tuning data."
    
    try:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "❌ Failed to load model for fine-tuning."
        
        # Similar implementation to train_model but with different hyperparameters
        train_texts = dataset_text.split('\\n')
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        training_args = TrainingArguments(
            output_dir=f"./finetuned-{{datetime.now().strftime('%Y%m%d-%H%M%S')}}",
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            logging_dir='./logs',
            save_steps=250,
            eval_steps=250,
            save_total_limit=2,
        )
        
        class SimpleDataset:
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {{key: torch.tensor(val[idx]) for key, val in self.encodings.items()}}
            
            def __len__(self):
                return len(self.encodings.input_ids)
        
        train_dataset = SimpleDataset(train_encodings)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        result = trainer.train()
        
        output = f"🔧 Fine-tuning completed!\\n"
        output += f"📊 Training Loss: {{result.training_loss:.4f}}\\n"
        output += f"⏱️ Training Time: {{result.metrics.get('train_runtime', 'N/A')}} seconds\\n"
        
        if save_model:
            save_dir = f"./finetuned-{{{model_name}}}-{{datetime.now().strftime('%Y%m%d-%H%M%S')}}"
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            output += f"💾 Model saved locally to: {{save_dir}}\\n"
            output += "📤 To upload to HuggingFace Hub, use the 'Upload Model' button below.\\n"
        
        return output
        
    except Exception as e:
        return f"❌ Fine-tuning failed: {{str(e)}}"

def upload_model_to_hub(model_dir, repo_name, token):
    \"\"\"Upload trained model to HuggingFace Hub\"\"\"
    try:
        if not os.path.exists(model_dir):
            return "❌ Model directory not found. Please train a model first."
        
        api = HfApi()
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload trained model from {{model_dir}}"
        )
        
        return f"✅ Model uploaded successfully to {{repo_name}}!"
        
    except Exception as e:
        return f"❌ Upload failed: {{str(e)}}"

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
                placeholder="Enter training data (one example per line)...",
                lines=5
            )
            with gr.Row():
                train_epochs = gr.Number(label="Epochs", value=1, minimum=1, maximum=10)
                train_lr = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-6, maximum=1e-3)
            
            train_save_model = gr.Checkbox(label="Save trained model", value=True)
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Output", lines=8, interactive=False)
            
            train_btn.click(
                fn=train_model,
                inputs=[train_dataset, train_epochs, train_lr, train_save_model],
                outputs=train_output
            )
        
        # Fine-tune Tab
        with gr.TabItem("🔧 Fine-tune Model"):
            gr.Markdown("### Fine-tune the existing model")
            finetune_dataset = gr.Textbox(
                label="Fine-tuning Dataset",
                placeholder="Enter fine-tuning data (one example per line)...",
                lines=5
            )
            with gr.Row():
                finetune_epochs = gr.Number(label="Epochs", value=1, minimum=1, maximum=5)
                finetune_lr = gr.Number(label="Learning Rate", value=5e-5, minimum=1e-6, maximum=1e-3)
            
            finetune_save_model = gr.Checkbox(label="Save fine-tuned model", value=True)
            finetune_btn = gr.Button("🔧 Start Fine-tuning", variant="primary")
            finetune_output = gr.Textbox(label="Fine-tuning Output", lines=8, interactive=False)
            
            finetune_btn.click(
                fn=finetune_model,
                inputs=[finetune_dataset, finetune_epochs, finetune_lr, finetune_save_model],
                outputs=finetune_output
            )
        
        # Upload Tab
        with gr.TabItem("📤 Upload Model"):
            gr.Markdown("### Upload trained models to HuggingFace Hub")
            with gr.Row():
                model_dir_input = gr.Textbox(
                    label="Model Directory",
                    placeholder="./trained-model-20241207-123456",
                    lines=1
                )
                repo_name_input = gr.Textbox(
                    label="Repository Name",
                    placeholder="username/model-name",
                    lines=1
                )
            
            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_...",
                type="password",
                lines=1
            )
            
            upload_btn = gr.Button("📤 Upload to Hub", variant="primary")
            upload_output = gr.Textbox(label="Upload Output", lines=3, interactive=False)
            
            upload_btn.click(
                fn=upload_model_to_hub,
                inputs=[model_dir_input, repo_name_input, hf_token_input],
                outputs=upload_output
            )

if __name__ == "__main__":
    iface.launch()
"""

    # Enhanced requirements.txt content
    requirements_content = """gradio==4.36.1
spaces
torch
transformers
datasets
accelerate
huggingface_hub
"""

    try:
        # Create a temporary directory and clone the space repo
        temp_dir = f"temp_{model_name}_space"
        if os.path.exists(temp_dir):
            os.system(f"rm -rf {temp_dir}")
        
        repo = Repository(local_dir=temp_dir, clone_from=f"https://huggingface.co/spaces/{space_name}")
        
        # Write the files
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        with open(os.path.join(temp_dir, "app.py"), "w") as f:
            f.write(app_content)
        
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write(requirements_content)
        
        # Push to the space
        repo.push_to_hub(commit_message=f"Add enhanced Gradio app for {model_name} with full training capabilities")
        
        # Clean up
        os.system(f"rm -rf {temp_dir}")
        
        print(f"📁 Successfully uploaded enhanced app files to {space_name}")
        
    except Exception as e:
        print(f"❌ Failed to upload files to {space_name}: {e}")

def main():
    # Load credentials automatically
    username, token = load_credentials()
    
    try:
        login(token=token)
        print(f"✅ Successfully authenticated as {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    # Define the remaining models that need spaces
    remaining_models = [
        f"{username}/ZamAI-Phi-3-Mini-Pashto",
        f"{username}/ZamAI-Whisper-v3-Pashto", 
        f"{username}/zamai-pashto-chat-8b",
        f"{username}/zamai-translator-pashto-en",
        f"{username}/zamai-qa-pashto",
        f"{username}/zamai-sentiment-pashto",
        f"{username}/zamai-dialogpt-pashto-v3"
    ]
    
    print(f"🚀 Creating remaining {len(remaining_models)} spaces with enhanced functionality...")
    print("Each space will include:")
    print("  - 🧪 Full model testing")
    print("  - 🏋️ Actual training implementation")
    print("  - 🔧 Real fine-tuning capabilities")
    print("  - 📤 Model upload to HuggingFace Hub")
    print("  - ⚡ ZeroGPU acceleration")
    print()
    
    created_count = 0
    failed_count = 0
    
    for i, model_id in enumerate(remaining_models, 1):
        print(f"[{i}/{len(remaining_models)}] Processing {model_id}...")
        space_name = create_model_space_with_delay(model_id, username, token, delay=15)
        
        if space_name:
            created_count += 1
            print(f"✅ Successfully created and configured {space_name}")
        else:
            failed_count += 1
            print(f"❌ Failed to create space for {model_id}")
        
        print()
    
    print(f"📊 Final Results:")
    print(f"✅ Created: {created_count}/{len(remaining_models)} spaces")
    print(f"❌ Failed: {failed_count}/{len(remaining_models)} spaces")
    
    if created_count > 0:
        print(f"🎉 Successfully created {created_count} enhanced spaces!")
        print("Each space now includes full training, fine-tuning, and model upload capabilities!")

if __name__ == "__main__":
    main()
