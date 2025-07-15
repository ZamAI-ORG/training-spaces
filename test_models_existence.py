import os
from pathlib import Path

# Optional: Uncomment if you want to try loading models with transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig

MODELS_DIR = Path("models")

# Key files to check for existence
KEY_FILES = [
    "config.json",
    "README.md",
    # Add more as needed, e.g. weights, tokenizer, etc.
]

# For some models, weights may be named differently
WEIGHT_FILES = [
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    # Add more formats if needed
]

def check_model_files(model_path):
    missing = []
    for fname in KEY_FILES:
        if not (model_path / fname).exists():
            missing.append(fname)
    # Check for at least one weight file
    if not any((model_path / wf).exists() for wf in WEIGHT_FILES):
        missing.append("weights file (e.g. pytorch_model.bin or model.safetensors)")
    return missing

# Try to load model with transformers
def try_load_model(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return True, None
    except Exception as e:
        return False, str(e)


def main():
    print(f"Checking models in: {MODELS_DIR.resolve()}")
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.'):
            continue
        print(f"\nModel: {model_dir.name}")
        missing = check_model_files(model_dir)
        if missing:
            print("  Missing files:")
            for m in missing:
                print(f"    - {m}")
        else:
            print("  All key files found.")
        # Optional: Try to load model
        success, err = try_load_model(model_dir)
        if success:
            print("  Model loaded successfully with transformers.")
        else:
            print(f"  Failed to load model: {err}")

if __name__ == "__main__":
    main()
