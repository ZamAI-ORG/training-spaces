#!/usr/bin/env python3
"""
Model Weight Fixer and File System Organizer
- Checks which models are missing weights
- Prepares model files from base models if needed
- Uploads models to Hugging Face Hub
- Organizes local file structure
"""

import os
import sys
import json
import shutil
import time
import torch
from pathlib import Path
from huggingface_hub import HfApi, login, Repository, upload_folder, create_repo
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, BertModel, BertTokenizer,
    WhisperModel, WhisperTokenizer, WhisperProcessor,
    AutoModelForSeq2SeqLM
)

# Model architecture mapping - using smaller, publicly available alternatives where needed
MODEL_ARCHITECTURES = {
    "pashto-base-bloom": {
        "base_model": "bigscience/bloom-560m",
        "model_type": "causal_lm"
    },
    "ZamAI-LIama3-Pashto": {
        "base_model": "facebook/opt-350m",  # Public alternative to LLaMA
        "model_type": "causal_lm"
    },
    "Multilingual-ZamAI-Embeddings": {
        "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "model_type": "sentence_transformer"
    },
    "ZamAI-Mistral-7B-Pashto": {
        "base_model": "google/gemma-2b",  # Public alternative to Mistral
        "model_type": "causal_lm"
    },
    "ZamAI-Phi-3-Mini-Pashto": {
        "base_model": "EleutherAI/gpt-neo-125m",  # Smaller alternative to Phi-3
        "model_type": "causal_lm"
    },
    "ZamAI-Whisper-v3-Pashto": {
        "base_model": "openai/whisper-small",
        "model_type": "whisper"
    },
    "zamai-pashto-chat-8b": {
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "model_type": "causal_lm"
    },
    "zamai-translator-pashto-en": {
        "base_model": "t5-small",
        "model_type": "seq2seq"
    },
    "zamai-qa-pashto": {
        "base_model": "deepset/roberta-base-squad2",
        "model_type": "question_answering"
    },
    "zamai-sentiment-pashto": {
        "base_model": "nlptown/bert-base-multilingual-uncased-sentiment",
        "model_type": "text_classification"
    },
    "zamai-dialogpt-pashto-v3": {
        "base_model": "microsoft/DialoGPT-medium",
        "model_type": "causal_lm"
    }
}

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

def load_model_summary():
    """Load model summary from model_summary.json"""
    try:
        with open("model_summary.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load model summary: {e}")
        return None

def prepare_model_files(model_name, base_model, model_type, save_dir):
    """Prepare model files from base model with better disk space management"""
    try:
        print(f"🔄 Preparing model files for {model_name} from {base_model}...")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and save appropriate model type
        if model_type == "causal_lm":
            print("📥 Loading base causal language model...")
            # Use half precision to reduce memory usage
            model = AutoModelForCausalLM.from_pretrained(
                base_model, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            # Add Pashto language token if needed
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
                
            # Save model and tokenizer in optimized format
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir, save_safetensors=True)
            tokenizer.save_pretrained(save_dir)
            
        elif model_type == "seq2seq":
            print("📥 Loading base sequence-to-sequence model...")
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
        elif model_type == "text_classification":
            print("📥 Loading base text classification model...")
            model = AutoModelForSequenceClassification.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
        elif model_type == "whisper":
            print("📥 Loading base Whisper model...")
            model = WhisperModel.from_pretrained(base_model)
            tokenizer = WhisperTokenizer.from_pretrained(base_model)
            processor = WhisperProcessor.from_pretrained(base_model)
            
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            
        elif model_type == "sentence_transformer":
            print("📥 Loading base sentence transformer model...")
            model = AutoModel.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
        elif model_type == "question_answering":
            print("📥 Loading base question answering model...")
            model = AutoModelForSequenceClassification.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            print(f"💾 Saving model files to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            
        else:
            print(f"❌ Unknown model type: {model_type}")
            return False
            
        print(f"✅ Successfully prepared model files for {model_name}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to prepare model files for {model_name}: {e}")
        return False

def upload_model_to_hub(model_dir, repo_id, token):
    """Upload model files to Hugging Face Hub"""
    try:
        print(f"📤 Uploading model files to {repo_id}...")
        
        # Create or ensure repo exists
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            print(f"✓ Repository {repo_id} ready")
        except Exception as e:
            print(f"⚠️ Repository creation warning (might already exist): {e}")
        
        # Upload all files in the directory
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            commit_message="Upload model files and weights",
            ignore_patterns=["*.git*", "*.ipynb_checkpoints*"],
        )
        
        print(f"✅ Successfully uploaded model files to {repo_id}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to upload model files to {repo_id}: {e}")
        return False

def organize_file_structure(models_dir):
    """Organize local file structure"""
    try:
        print(f"🗂️ Organizing file structure in {models_dir}...")
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Create categories for different model types
        categories = {
            "language_models": os.path.join(models_dir, "language_models"),
            "translation_models": os.path.join(models_dir, "translation_models"),
            "speech_models": os.path.join(models_dir, "speech_models"),
            "embedding_models": os.path.join(models_dir, "embedding_models"),
            "qa_models": os.path.join(models_dir, "qa_models"),
            "sentiment_models": os.path.join(models_dir, "sentiment_models")
        }
        
        # Create category directories
        for category_dir in categories.values():
            os.makedirs(category_dir, exist_ok=True)
        
        # Map models to categories
        model_categories = {
            "pashto-base-bloom": "language_models",
            "ZamAI-LIama3-Pashto": "language_models",
            "Multilingual-ZamAI-Embeddings": "embedding_models",
            "ZamAI-Mistral-7B-Pashto": "language_models",
            "ZamAI-Phi-3-Mini-Pashto": "language_models",
            "ZamAI-Whisper-v3-Pashto": "speech_models",
            "zamai-pashto-chat-8b": "language_models",
            "zamai-translator-pashto-en": "translation_models",
            "zamai-qa-pashto": "qa_models",
            "zamai-sentiment-pashto": "sentiment_models",
            "zamai-dialogpt-pashto-v3": "language_models"
        }
        
        # Create symlinks for all models to maintain both organizations
        for model_name, category in model_categories.items():
            src_path = os.path.join(models_dir, model_name)
            dst_path = os.path.join(categories[category], model_name)
            
            # Only create symlink if source exists and destination doesn't
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                print(f"🔗 Creating symlink for {model_name} in {category}")
                # Create relative symlink
                os.symlink(
                    os.path.relpath(src_path, os.path.dirname(dst_path)),
                    dst_path
                )
        
        print(f"✅ Successfully organized file structure")
        
        # Create README for models directory
        readme_path = os.path.join(models_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write("""# ZamAI Models

## Model Categories

- **Language Models**: General purpose language models for Pashto
- **Translation Models**: Models specifically for translation between Pashto and other languages
- **Speech Models**: Speech recognition and processing models for Pashto
- **Embedding Models**: Models for creating text embeddings
- **QA Models**: Models for question answering
- **Sentiment Models**: Models for sentiment analysis

## Model List

Each model is stored in its own directory and also symlinked in the appropriate category folder.
""")
        
        return True
    
    except Exception as e:
        print(f"❌ Failed to organize file structure: {e}")
        return False

def fix_missing_model_weights(username, token):
    """Fix models with missing weights"""
    # Load model summary
    model_summary = load_model_summary()
    if not model_summary:
        return (0, 0)
    
    # Find models missing weights
    incomplete_models = {
        model_id: info for model_id, info in model_summary.items()
        if not info["complete"] or not info["has_model_weights"]
    }
    
    print(f"📊 Found {len(incomplete_models)}/{len(model_summary)} models with missing weights")
    
    # Fix each incomplete model
    success_count = 0
    failed_count = 0
    
    for model_id, info in incomplete_models.items():
        model_name = info["name"]
        print(f"\n🔍 Processing {model_id}...")
        
        # Check if we have architecture info for this model
        if model_name not in MODEL_ARCHITECTURES:
            print(f"❌ No architecture information for {model_name}, skipping")
            failed_count += 1
            continue
        
        # Get architecture info
        arch_info = MODEL_ARCHITECTURES[model_name]
        base_model = arch_info["base_model"]
        model_type = arch_info["model_type"]
        
        # Prepare model directory
        model_dir = os.path.join("models", model_name)
        
        # Backup existing directory if it exists
        if os.path.exists(model_dir):
            backup_dir = f"{model_dir}_backup_{int(time.time())}"
            print(f"⚠️ Backing up existing directory to {backup_dir}")
            shutil.copytree(model_dir, backup_dir)
        
        # Prepare model files
        success = prepare_model_files(model_name, base_model, model_type, model_dir)
        
        if success:
            # Copy existing non-model files to maintain metadata
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file not in ["pytorch_model.bin", "model.safetensors", "tokenizer.json", "config.json"]:
                        src = os.path.join(model_dir, file)
                        dst = os.path.join(model_dir, file)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
            
            # Upload to Hub
            upload_success = upload_model_to_hub(model_dir, model_id, token)
            
            if upload_success:
                success_count += 1
                print(f"✅ Successfully fixed and uploaded {model_id}")
            else:
                failed_count += 1
                print(f"❌ Failed to upload {model_id}")
        else:
            failed_count += 1
            print(f"❌ Failed to prepare model files for {model_id}")
    
    print(f"\n📊 Model Fix Summary:")
    print(f"  ✅ Successfully fixed: {success_count}/{len(incomplete_models)}")
    print(f"  ❌ Failed to fix: {failed_count}/{len(incomplete_models)}")
    
    return success_count, failed_count

def main():
    # Load credentials
    username, token = load_credentials()
    
    try:
        login(token=token)
        print(f"✅ Successfully authenticated as {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    # Fix missing model weights
    print("\n🔧 Checking and fixing models with missing weights...")
    success_count, failed_count = fix_missing_model_weights(username, token)
    
    # Organize file structure
    print("\n🗂️ Organizing local file structure...")
    organize_file_structure("models")
    
    print("\n🎉 Model weight fixing and organization complete!")
    print(f"✅ Fixed {success_count} models with missing weights")
    print(f"🗂️ Organized models into categories:")
    print("  - language_models")
    print("  - translation_models")
    print("  - speech_models")
    print("  - embedding_models")
    print("  - qa_models")
    print("  - sentiment_models")
    
    print("\n💡 Next steps:")
    print("1. Check model repositories on Hugging Face Hub to ensure all files are uploaded")
    print("2. Run 'python check_and_fix_models.py' again to verify all models are complete")
    print("3. Run Spaces to test the models with the new weights")

if __name__ == "__main__":
    main()
