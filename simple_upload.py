#!/usr/bin/env python3
"""
Simple script to upload model and space
"""

import os
import sys
from huggingface_hub import HfApi, login, create_repo, upload_folder

def main():
    try:
        # Model name
        model_name = "ZamAI-Mistral-7B-Pashto"
        
        # Get credentials
        with open("HF-Credentials.txt", "r") as f:
            lines = f.readlines()
        username = lines[0].split(":")[1].strip().strip("<>")
        token = lines[1].split(":")[1].strip().strip("<>")
        
        print(f"Username: {username}")
        print(f"Token: {token[:5]}...{token[-5:]}")
        
        # Login
        print("Logging in...")
        login(token=token)
        print("Login successful")
        
        # Upload model
        model_id = f"{username}/{model_name}"
        model_dir = f"models/{model_name}"
        print(f"Uploading model {model_id} from {model_dir}")
        
        # Create repo
        print("Creating model repo...")
        create_repo(repo_id=model_id, repo_type="model", token=token, exist_ok=True)
        print("Model repo created")
        
        # Upload model files
        print("Uploading model files...")
        upload_folder(
            folder_path=model_dir,
            repo_id=model_id,
            repo_type="model",
            token=token,
            ignore_patterns=["*.git*", "__pycache__", "*.pyc"]
        )
        print("Model files uploaded")
        
        # Upload space
        space_id = f"{username}/{model_name}-space"
        space_dir = f"spaces/{model_name}-space"
        print(f"Uploading space {space_id} from {space_dir}")
        
        # Create space repo
        print("Creating space repo...")
        create_repo(repo_id=space_id, repo_type="space", token=token, exist_ok=True, space_sdk="gradio")
        print("Space repo created")
        
        # Upload space files
        print("Uploading space files...")
        upload_folder(
            folder_path=space_dir,
            repo_id=space_id,
            repo_type="space",
            token=token,
            ignore_patterns=["*.git*", "__pycache__", "*.pyc"]
        )
        print("Space files uploaded")
        
        print("All operations completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
