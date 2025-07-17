import sys
import os
from huggingface_hub import HfApi, login, create_repo, delete_repo, Repository
import requests

def setup_hf_auth():
    """Set up Hugging Face authentication using a token from a file."""
    try:
        with open("HF-Credentials.txt", "r") as f:
            token = f.read().strip()
        if not token:
            print("❌ No token found in HF-Credentials.txt. Please create the file and add your token.")
            sys.exit(1)
        
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face")
        return token
    except FileNotFoundError:
        print("❌ HF-Credentials.txt not found. Please create it and add your Hugging Face token.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
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

def delete_space(space_id, token):
    try:
        delete_repo(repo_id=space_id, repo_type="space", token=token)
        print(f"✅ Deleted space: {space_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to delete space {space_id}: {e}")
        return False

def create_model_space(model_id, username, token):
    model_name = model_id.split('/')[-1]
    space_name = f"{username}/{model_name}-space"
    try:
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
        
        return space_name
    except Exception as e:
        print(f"❌ Failed to create space {space_name}: {e}")
        return None

def create_space_files(space_name, model_id, model_name, token):
    """Create and upload app.py, requirements.txt, and README.md to the space"""
    
    # README.md content
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

    # app.py content with ZeroGPU decorator
    app_content = f"""import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

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

def train_model(dataset_text, epochs=1, learning_rate=2e-5):
    \"\"\"Train the model (placeholder implementation)\"\"\"
    return f"🚀 Training started with {{epochs}} epochs and learning rate {{learning_rate}}\\n\\nNote: This is a placeholder. Actual training requires dataset preparation and more computational resources."

def finetune_model(dataset_text, epochs=1, learning_rate=5e-5):
    \"\"\"Fine-tune the model (placeholder implementation)\"\"\"
    return f"🔧 Fine-tuning started with {{epochs}} epochs and learning rate {{learning_rate}}\\n\\nNote: This is a placeholder. Actual fine-tuning requires dataset preparation and more computational resources."

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
"""

    # requirements.txt content
    requirements_content = """gradio
spaces
torch
transformers
datasets
accelerate
sentencepiece
protobuf
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
        repo.push_to_hub(commit_message=f"Add Gradio app for {model_name} with train/finetune/test")
        
        # Clean up
        os.system(f"rm -rf {temp_dir}")
        
        print(f"📁 Successfully uploaded app files to {space_name}")
        
    except Exception as e:
        print(f"❌ Failed to upload files to {space_name}: {e}")

def main():
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ No username provided. Exiting.")
        sys.exit(1)
    token = setup_hf_auth()

    print("🔍 Fetching your models from Hugging Face Hub...")
    model_ids = list_user_models(username, token)
    if not model_ids:
        print("❌ No models found on your Hugging Face account.")
        sys.exit(1)
    print(f"📦 Found {len(model_ids)} models:")
    for m in model_ids:
        print(f"  - {m}")

    print("\n🔍 Fetching your existing Spaces...")
    spaces = list_user_spaces(username, token)
    if spaces:
        print(f"Found {len(spaces)} spaces.")
        delete_input = input("Do you want to delete all existing spaces? (yes/no): ").strip().lower()
        if delete_input == 'yes':
            print("Deleting spaces...")
            for space in spaces:
                delete_space(space, token)
        else:
            print("Skipping deletion of spaces.")
    else:
        print("No spaces to delete.")

    print("\n🚀 Creating one Space per model...")
    for model_id in model_ids:
        create_model_space(model_id, username, token)
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
