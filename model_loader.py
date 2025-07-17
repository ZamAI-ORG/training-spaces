import os
import sys
from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoTokenizer

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

def load_and_save_models(username, token):
    """
    Loads models from the Hugging Face Hub, saves them to local directories,
    and pushes them back to the Hub to ensure they are up-to-date.
    """
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"❌ Models directory '{models_dir}' not found.")
        return

    model_names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    api = HfApi()

    for model_name in model_names:
        model_id = f"{username}/{model_name}"
        local_model_path = os.path.join(models_dir, model_name)
        
        print(f"Processing model: {model_id}")

        try:
            # 1. Load model and tokenizer from Hub
            print(f"  - Downloading model and tokenizer for {model_id}...")
            # Use AutoModel for a generic model type, adjust if you have specific model types
            model = AutoModel.from_pretrained(model_id, token=token)
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

            # 2. Save them to the local directory
            print(f"  - Saving model and tokenizer to {local_model_path}...")
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            
            # 3. Push the complete model to the Hub
            print(f"  - Uploading {local_model_path} to {model_id}...")
            api.upload_folder(
                folder_path=local_model_path,
                repo_id=model_id,
                repo_type="model",
                commit_message=f"Sync local model files for {model_name}"
            )
            print(f"✅ Successfully synced {model_id}")

        except Exception as e:
            print(f"❌ Failed to process {model_id}: {e}")

def main():
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ No username provided. Exiting.")
        sys.exit(1)
    
    token = setup_hf_auth()
    load_and_save_models(username, token)
    print("\n✅ All models processed.")

if __name__ == "__main__":
    main()
