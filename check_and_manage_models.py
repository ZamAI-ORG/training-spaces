import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from huggingface_hub import model_info, HfApi, hf_hub_download
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face Hub API
api = HfApi()
USERNAME = "tasal9"  # Your HF Hub username

# Sample inputs for testing different models
SAMPLES = {
    "general": "سلام، څنګه یی؟",  # "Hello, how are you?" in Pashto
    "qa": "افغانستان په کوم قاره کې دی؟",  # "In which continent is Afghanistan?" in Pashto
    "translation": "زه غواړم د ژبې زده کړه وکړم",  # "I want to learn a language" in Pashto
    "sentiment": "دا خبره ډيره ښه ده",  # "This is very good" in Pashto
}

# List of all models from the models directory
MODEL_DIRS = [
    d for d in os.listdir("models") 
    if os.path.isdir(os.path.join("models", d)) and not d.endswith("backup")
]

# Dict to map model types to appropriate model classes and test inputs
MODEL_TYPES = {
    "chat": {"class": AutoModelForCausalLM, "sample": "general"},
    "qa": {"class": AutoModelForCausalLM, "sample": "qa"},
    "translation": {"class": AutoModelForCausalLM, "sample": "translation"},
    "sentiment": {"class": AutoModelForCausalLM, "sample": "sentiment"},
    "embeddings": {"class": AutoModel, "sample": "general"},
    "whisper": {"class": AutoModelForCausalLM, "sample": "general"},
}

# Map model folder names to their types
MODEL_TYPE_MAPPING = {
    model_name: next(
        (t for t in MODEL_TYPES if t in model_name.lower()), 
        "chat"  # Default type if no match
    )
    for model_name in MODEL_DIRS
}

def check_model_on_hub(model_name):
    """Check if model exists on Hugging Face Hub and has weights."""
    repo_name = f"{USERNAME}/{model_name}"
    logger.info(f"Checking HF Hub for model: {repo_name}")
    try:
        info = model_info(repo_name)
        siblings = info.siblings if info.siblings is not None else []
        has_weights = any(
            f.rfilename.endswith(".bin") or f.rfilename.endswith(".safetensors")
            for f in siblings
        )
        if has_weights:
            logger.info(f"✅ Model weights found for {repo_name} on HF Hub.")
            return True
        else:
            logger.info(f"❌ No model weights found for {repo_name} on HF Hub.")
            return False
    except Exception as e:
        logger.error(f"Error checking {repo_name}: {e}")
        return False

def test_model(model_name, model_type):
    """Test a model by loading it and running inference."""
    repo_name = f"{USERNAME}/{model_name}"
    local_path = os.path.join("models", model_name)
    model_class = MODEL_TYPES[model_type]["class"]
    sample_key = MODEL_TYPES[model_type]["sample"]
    sample_input = SAMPLES[sample_key]
    
    logger.info(f"Testing model: {repo_name}")
    try:
        # Try loading tokenizer and model from Hub
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        model = model_class.from_pretrained(repo_name)
        logger.info(f"Model loaded successfully from Hub.")
        
        # Test model if it's not an embedding model
        if model_class != AutoModel:  # Skip testing for embedding models
            inputs = tokenizer(sample_input, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=100)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Sample output: {result}")
        else:
            # For embedding models
            inputs = tokenizer(sample_input, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state
            logger.info(f"Embedding shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing {repo_name}: {e}")
        return False

def upload_model_to_hub(model_name, model_type):
    """Upload model to Hugging Face Hub if not already there."""
    repo_name = f"{USERNAME}/{model_name}"
    local_path = os.path.join("models", model_name)
    
    logger.info(f"Preparing to upload {model_name} to Hub")
    
    # Check if we have local files to upload
    if not os.path.exists(local_path) or not os.listdir(local_path):
        logger.error(f"No local files found for {model_name}. Can't upload.")
        return False
    
    # Check for minimum required files
    has_config = os.path.exists(os.path.join(local_path, "config.json"))
    if not has_config:
        logger.error(f"Missing config.json for {model_name}. Can't upload.")
        return False
    
    # Create or update repo
    try:
        api.create_repo(repo_id=repo_name, exist_ok=True)
        logger.info(f"Repository {repo_name} created/confirmed on Hub")
        
        # Upload all files in the directory
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            commit_message=f"Upload model files for {model_name}"
        )
        logger.info(f"✅ Files uploaded to {repo_name} successfully")
        return True
    except Exception as e:
        logger.error(f"Error uploading {model_name} to Hub: {e}")
        return False

def create_readme(model_name, model_type):
    """Create an improved README for the model."""
    repo_name = f"{USERNAME}/{model_name}"
    local_path = os.path.join("models", model_name)
    readme_path = os.path.join(local_path, "README.md")
    
    # Check if README already exists
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            current_readme = f.read()
        logger.info(f"Existing README found for {model_name}, updating...")
    else:
        current_readme = ""
        logger.info(f"No README found for {model_name}, creating new one...")
    
    # Model type descriptions
    type_descriptions = {
        "chat": "a conversational AI model fine-tuned for Pashto language dialogue",
        "qa": "a question-answering model trained on Pashto language data",
        "translation": "a neural machine translation model for Pashto-English translation",
        "sentiment": "a sentiment analysis model for Pashto text",
        "embeddings": "a text embedding model trained on multilingual data including Pashto",
        "whisper": "a speech recognition model fine-tuned for Pashto language",
    }
    
    # Create improved README content
    model_type_desc = type_descriptions.get(model_type, "a language model")
    readme_content = f"""# {model_name}

## Model Description

{model_name} is {model_type_desc} developed by ZamAI. It is designed to provide natural language processing capabilities for the Pashto language.

## Features

- Optimized for Pashto language text
- {'Conversational capabilities' if 'chat' in model_type else 'Specialized for ' + model_type + ' tasks'}
- Developed as part of the ZamAI initiative to improve NLP tools for lower-resource languages

## Usage

```python
from transformers import AutoTokenizer, {'AutoModelForCausalLM' if model_type != 'embeddings' else 'AutoModel'}

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = {'AutoModelForCausalLM' if model_type != 'embeddings' else 'AutoModel'}.from_pretrained("{repo_name}")

# Example usage
{'# For text generation models' if model_type != 'embeddings' else '# For embedding models'}
text = "سلام، څنګه یی؟"  # "Hello, how are you?" in Pashto
{'inputs = tokenizer(text, return_tensors="pt")\n' if model_type != 'embeddings' else 'inputs = tokenizer(text, return_tensors="pt")\n'}
{'with torch.no_grad():\n    outputs = model.generate(**inputs, max_length=100)\nresult = tokenizer.decode(outputs[0], skip_special_tokens=True)\nprint(result)' if model_type != 'embeddings' else 'with torch.no_grad():\n    embeddings = model(**inputs).last_hidden_state\nprint(embeddings.shape)'}
```

## Training

This model {'was fine-tuned' if 'base' not in model_name.lower() else 'was trained'} on {'Pashto language texts' if model_type in ['chat', 'embeddings'] else 'a dataset of Pashto ' + model_type + ' examples'}.

## License

Please see the license file in the repository for usage rights and limitations.

## Acknowledgments

This model is part of the ZamAI initiative to improve natural language processing for lower-resource languages.
"""
    
    # Write the new README
    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"README created/updated for {model_name}")
    
    return True

def check_datasets():
    """Check datasets in the repository and on Hugging Face Hub."""
    local_datasets = [f for f in os.listdir("datasets") if f.endswith('.json')]
    logger.info(f"Found {len(local_datasets)} local datasets: {local_datasets}")
    
    # Check each dataset on the Hub
    for dataset in local_datasets:
        dataset_name = dataset.replace('.json', '')
        repo_name = f"{USERNAME}/{dataset_name}"
        
        try:
            # Check if dataset exists on Hub
            response = requests.get(f"https://huggingface.co/api/datasets/{repo_name}")
            if response.status_code == 200:
                logger.info(f"✅ Dataset {repo_name} found on Hub")
            else:
                logger.info(f"❌ Dataset {repo_name} not found on Hub")
                # Read local dataset file to get stats
                dataset_path = os.path.join("datasets", dataset)
                try:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"Dataset {dataset} has {len(data)} records")
                    elif isinstance(data, dict):
                        logger.info(f"Dataset {dataset} structure: {list(data.keys())}")
                except Exception as e:
                    logger.error(f"Error reading dataset {dataset}: {e}")
        except Exception as e:
            logger.error(f"Error checking dataset {dataset_name}: {e}")

def determine_training_needs():
    """Determine which models need training or fine-tuning."""
    training_needs = {}
    
    for model_name in MODEL_DIRS:
        local_path = os.path.join("models", model_name)
        model_type = MODEL_TYPE_MAPPING.get(model_name, "chat")
        
        # Check if model has weights locally
        has_local_weights = (
            os.path.exists(os.path.join(local_path, "pytorch_model.bin")) or 
            os.path.exists(os.path.join(local_path, "model.safetensors"))
        )
        
        # Check if model exists on Hub
        has_hub_weights = check_model_on_hub(model_name)
        
        if not has_local_weights and not has_hub_weights:
            training_needs[model_name] = "Training required - no weights found"
        elif not has_hub_weights:
            training_needs[model_name] = "Upload to Hub required - local weights exist"
        else:
            # Test model performance to determine if more training is needed
            test_model(model_name, model_type)
            training_needs[model_name] = "Model exists, fine-tuning may improve performance"
    
    # Save training needs to file
    with open("model_training_needs.json", "w") as f:
        json.dump(training_needs, f, indent=4)
    
    logger.info("Training needs assessment completed and saved to model_training_needs.json")
    return training_needs

def prepare_for_training(model_name):
    """Prepare a model for training/fine-tuning."""
    local_path = os.path.join("models", model_name)
    model_type = MODEL_TYPE_MAPPING.get(model_name, "chat")
    
    # Create training configuration file
    training_config = {
        "model_name": model_name,
        "model_type": model_type,
        "training_data": f"datasets/pashto_{model_type}.json" if f"pashto_{model_type}.json" in os.listdir("datasets") else "datasets/pashto_chat.json",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "max_length": 512
    }
    
    # Save the training config
    config_path = os.path.join(local_path, "training_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=4)
    
    logger.info(f"Training configuration created for {model_name}")
    return True

def main():
    """Main function to process all models."""
    logger.info("Starting model checking process")
    
    # Check all models on Hub
    model_status = {}
    for model_name in MODEL_DIRS:
        model_type = MODEL_TYPE_MAPPING.get(model_name, "chat")
        model_status[model_name] = {
            "exists_on_hub": check_model_on_hub(model_name),
            "model_type": model_type
        }
    
    # Log summary of model statuses
    logger.info("\n===== MODEL STATUS SUMMARY =====")
    for model, status in model_status.items():
        logger.info(f"{model}: {'✅ Found on Hub' if status['exists_on_hub'] else '❌ Not found on Hub'} (Type: {status['model_type']})")
    
    # Test models that exist
    for model_name, status in model_status.items():
        if status["exists_on_hub"]:
            test_result = test_model(model_name, status["model_type"])
            model_status[model_name]["test_successful"] = test_result
    
    # Create/update READMEs
    for model_name, status in model_status.items():
        create_readme(model_name, status["model_type"])
    
    # Upload models that don't exist on Hub
    for model_name, status in model_status.items():
        if not status["exists_on_hub"]:
            upload_model_to_hub(model_name, status["model_type"])
    
    # Prepare models for training
    for model_name in MODEL_DIRS:
        prepare_for_training(model_name)
    
    # Determine training needs
    training_needs = determine_training_needs()
    
    # Check datasets
    check_datasets()
    
    logger.info("All processing completed!")
    return model_status, training_needs

if __name__ == "__main__":
    main()
