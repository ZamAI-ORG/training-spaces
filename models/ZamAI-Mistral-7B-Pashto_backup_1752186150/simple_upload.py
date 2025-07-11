#!/usr/bin/env python3
"""
Simple ZamAI Mistral Model Upload Script
"""

import os
import sys
from huggingface_hub import HfApi, create_repo, upload_folder

def main():
    print("🚀 Starting ZamAI Mistral model upload...")
    
    # Check if HF token exists
    hf_token_path = "../Multilingual-ZamAI-Embeddings/HF-Token.txt"
    if not os.path.exists(hf_token_path):
        print("❌ HF Token file not found")
        return False
    
    # Read token
    with open(hf_token_path, 'r') as f:
        token = f.read().strip()
    
    print("✅ Token loaded successfully")
    
    # Initialize API
    api = HfApi(token=token)
    repo_id = "tasal9/ZamAI-Mistral-7B-Pashto"
    
    try:
        # Create repo
        print("📁 Creating repository...")
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print("✅ Repository ready")
        
        # Upload files
        print("📤 Uploading files...")
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload ZamAI Mistral model repository",
            ignore_patterns=[".git/", "__pycache__/", "*.pyc"]
        )
        
        print(f"🎉 Successfully uploaded to https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
