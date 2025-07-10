#!/usr/bin/env python3
"""
Comprehensive Model Management Script
- Load models from HuggingFace Hub
- Save models locally and to repos
- Manage model versions
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from huggingface_hub import HfApi, login, create_repo, upload_folder, download
from datetime import datetime
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

def list_user_models(username, token):
    """List all models for the user"""
    api = HfApi()
    try:
        models = api.list_models(author=username, token=token)
        return [m.id for m in models]
    except Exception as e:
        print(f"❌ Failed to list models: {e}")
        return []

def download_model_locally(model_id, local_dir=None):
    """Download a model from HuggingFace Hub to local directory"""
    try:
        if local_dir is None:
            model_name = model_id.split('/')[-1]
            local_dir = f"./models/{model_name}"
        
        print(f"📥 Downloading {model_id} to {local_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model files
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Save locally
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        
        # Save model info
        model_info = {
            "model_id": model_id,
            "download_date": datetime.now().isoformat(),
            "local_path": local_dir,
            "model_type": model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
        }
        
        with open(os.path.join(local_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Successfully downloaded {model_id} to {local_dir}")
        return local_dir
        
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        return None

def upload_model_to_repo(local_dir, repo_name, token, private=False):
    """Upload a local model to HuggingFace Hub"""
    try:
        print(f"📤 Uploading {local_dir} to {repo_name}...")
        
        # Create repository
        api = HfApi()
        try:
            create_repo(repo_id=repo_name, token=token, private=private, exist_ok=True)
            print(f"✅ Repository {repo_name} created/verified")
        except Exception as e:
            print(f"⚠️  Repository creation warning: {e}")
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload model from {local_dir}"
        )
        
        print(f"✅ Successfully uploaded to {repo_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload {local_dir} to {repo_name}: {e}")
        return False

def backup_all_models(username, token):
    """Download and backup all user models locally"""
    print(f"🔍 Getting list of all models for {username}...")
    model_ids = list_user_models(username, token)
    
    if not model_ids:
        print("❌ No models found")
        return
    
    print(f"📦 Found {len(model_ids)} models to backup:")
    for m in model_ids:
        print(f"  - {m}")
    
    backup_dir = f"./model_backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    successful_backups = 0
    failed_backups = 0
    
    for i, model_id in enumerate(model_ids, 1):
        print(f"\n[{i}/{len(model_ids)}] Backing up {model_id}...")
        model_name = model_id.split('/')[-1]
        local_path = os.path.join(backup_dir, model_name)
        
        result = download_model_locally(model_id, local_path)
        if result:
            successful_backups += 1
        else:
            failed_backups += 1
    
    print(f"\n📊 Backup Summary:")
    print(f"✅ Successful: {successful_backups}")
    print(f"❌ Failed: {failed_backups}")
    print(f"📁 Backup location: {backup_dir}")

def clone_model_to_new_repo(source_model_id, new_repo_name, username, token):
    """Clone a model to a new repository"""
    try:
        print(f"🔄 Cloning {source_model_id} to {new_repo_name}...")
        
        # Download the source model
        temp_dir = f"./temp_clone_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        local_path = download_model_locally(source_model_id, temp_dir)
        
        if not local_path:
            return False
        
        # Upload to new repository
        full_repo_name = f"{username}/{new_repo_name}" if '/' not in new_repo_name else new_repo_name
        success = upload_model_to_repo(local_path, full_repo_name, token)
        
        # Cleanup temp directory
        os.system(f"rm -rf {temp_dir}")
        
        if success:
            print(f"✅ Successfully cloned {source_model_id} to {full_repo_name}")
        
        return success
        
    except Exception as e:
        print(f"❌ Failed to clone model: {e}")
        return False

def main():
    username, token = load_credentials()
    
    try:
        login(token=token)
        print(f"✅ Successfully authenticated as {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    print("\n🤖 Model Management Menu:")
    print("1. List all my models")
    print("2. Download a specific model locally")
    print("3. Upload a local model to new repo")
    print("4. Backup all models locally")
    print("5. Clone model to new repository")
    print("6. Exit")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    if choice == "1":
        print("\n🔍 Fetching your models...")
        models = list_user_models(username, token)
        print(f"\n📦 Found {len(models)} models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    
    elif choice == "2":
        model_id = input("Enter model ID (e.g., tasal9/model-name): ").strip()
        local_dir = input("Enter local directory (optional, press Enter for default): ").strip()
        local_dir = local_dir if local_dir else None
        download_model_locally(model_id, local_dir)
    
    elif choice == "3":
        local_dir = input("Enter local model directory: ").strip()
        repo_name = input("Enter repository name (e.g., username/model-name): ").strip()
        private = input("Make repository private? (y/N): ").strip().lower() == 'y'
        upload_model_to_repo(local_dir, repo_name, token, private)
    
    elif choice == "4":
        confirm = input(f"This will download all models for {username}. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            backup_all_models(username, token)
        else:
            print("❌ Backup cancelled")
    
    elif choice == "5":
        source_id = input("Enter source model ID: ").strip()
        new_name = input("Enter new repository name: ").strip()
        clone_model_to_new_repo(source_id, new_name, username, token)
    
    elif choice == "6":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()
