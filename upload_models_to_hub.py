#!/usr/bin/env python3
"""
Script to upload models to Hugging Face Hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login, create_repo
import argparse

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

def upload_model(model_path, repo_name, token, private=False):
    """Upload a model to Hugging Face Hub"""
    api = HfApi()
    
    try:
        # Check if model directory has content
        if not any(model_path.iterdir()):
            print(f"⚠️  Skipping {repo_name}: Directory is empty")
            return False
            
        print(f"📤 Uploading {repo_name}...")
        
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
            print(f"✅ Repository {repo_name} created/verified")
        except Exception as e:
            print(f"⚠️  Repository creation warning for {repo_name}: {e}")
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload {repo_name} model"
        )
        
        print(f"✅ Successfully uploaded {repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {repo_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--models-dir", default="./models", help="Directory containing models")
    parser.add_argument("--private", action="store_true", help="Make repositories private")
    parser.add_argument("--username", help="Hugging Face username (will be prompted if not provided)")
    
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
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory {models_dir} does not exist")
        sys.exit(1)
    
    print(f"🔍 Scanning models in {models_dir}")
    
    # Get all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("❌ No model directories found")
        sys.exit(1)
    
    print(f"📦 Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")
    
    proceed = input("\nProceed with upload? (y/N): ").strip().lower()
    if proceed != 'y':
        print("❌ Upload cancelled")
        sys.exit(0)
    
    # Upload each model
    successful_uploads = 0
    failed_uploads = 0
    
    for model_dir in model_dirs:
        repo_name = f"{username}/{model_dir.name}"
        success = upload_model(model_dir, repo_name, token, args.private)
        
        if success:
            successful_uploads += 1
        else:
            failed_uploads += 1
    
    print(f"\n📊 Upload Summary:")
    print(f"✅ Successful: {successful_uploads}")
    print(f"❌ Failed: {failed_uploads}")
    print(f"📝 Total: {len(model_dirs)}")

if __name__ == "__main__":
    main()
