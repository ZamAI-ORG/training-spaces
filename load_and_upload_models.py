import sys
import os
from huggingface_hub import HfApi, login, Repository
from transformers import AutoModel, AutoTokenizer

def setup_hf_auth():
    """Set up Hugging Face authentication using an environment variable."""
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("❌ HUGGING_FACE_HUB_TOKEN environment variable not set.")
        print("Please set it to your Hugging Face token.")
        sys.exit(1)
    
    try:
        login(token=token)
        print("✅ Successfully authenticated with Hugging Face")
        return token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

def download_and_save_model(model_id, local_dir):
    """Downloads a model and its tokenizer from the Hub and saves it locally."""
    print(f"⬇️ Downloading model: {model_id}")
    try:
        # Create the target directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # Download and save the model and tokenizer
        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        model.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        
        print(f"✅ Successfully saved model and tokenizer to {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download or save model {model_id}: {e}")
        return False

def upload_model_to_hub(model_id, local_dir, username):
    """Uploads a local model directory to the Hugging Face Hub."""
    print(f"⬆️ Uploading model: {model_id} to your Hub account.")
    try:
        api = HfApi()
        repo_id = f"{username}/{model_id.split('/')[-1]}"
        
        api.create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model"
        )

        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add/update model weights for {model_id}"
        )
        print(f"✅ Successfully uploaded {model_id} to {repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload model {model_id}: {e}")

def main():
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ No username provided. Exiting.")
        sys.exit(1)
    
    token = setup_hf_auth()
    
    models_dir = "models"
    if not os.path.isdir(models_dir):
        print(f"❌ Models directory '{models_dir}' not found.")
        sys.exit(1)
        
    local_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    
    print(f"Found {len(local_models)} local model directories.")

    for model_name in local_models:
        model_id = f"{username}/{model_name}"
        local_dir = os.path.join(models_dir, model_name)
        
        # Check if the directory is empty
        if not os.listdir(local_dir):
            print(f"Directory for {model_name} is empty. Downloading from Hub...")
            if download_and_save_model(model_id, local_dir):
                upload_model_to_hub(model_id, local_dir, username)
        else:
            print(f"Directory for {model_name} is not empty. Assuming it's complete and uploading.")
            upload_model_to_hub(model_id, local_dir, username)

    print("\n✅ All models processed.")

if __name__ == "__main__":
    main()
