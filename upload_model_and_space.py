#!/usr/bin/env python3
"""
Script to upload model weights and configure space
"""

import os
import sys
from huggingface_hub import HfApi, login, create_repo, Repository, upload_folder
import json

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

def upload_model_weights(model_name, username, token):
    """Upload model weights to Hugging Face Hub"""
    model_id = f"{username}/{model_name}"
    model_dir = f"models/{model_name}"
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory {model_dir} not found")
        return False
    
    print(f"📤 Uploading model weights from {model_dir} to {model_id}...")
    
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
            folder_path=model_dir,
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
    model_name = "ZamAI-Mistral-7B-Pashto"
    
    try:
        # Load credentials
        print("Loading credentials...")
        username, token = load_credentials()
        print(f"Username: {username}")
        
        # Login to Hugging Face
        print("Logging in to Hugging Face...")
        login(token=token)
        print("Login successful")
        
        # Upload model weights
        print("Uploading model weights...")
        success = upload_model_weights(model_name, username, token)
        if success:
            # Update model summary
            print("Updating model summary...")
            update_model_summary(model_name, username, token)
        
        # Upload space
        print("Uploading space...")
        upload_space(model_name, username, token)
        
        print("All operations completed successfully!")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
