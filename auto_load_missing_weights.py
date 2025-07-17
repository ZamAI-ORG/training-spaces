import os
import json
from pathlib import Path
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
    AutoModelForTokenClassification, AutoModelForSpeechSeq2Seq, AutoConfig
)

def find_base_model_id(model_dir):
    # Try to extract base_model from README.md
    readme_path = model_dir / "README.md"
    if readme_path.exists():
        with open(readme_path) as f:
            for line in f:
                if line.strip().startswith("base_model:"):
                    base_model_id = line.split(":", 1)[1].strip()
                    if base_model_id:
                        return base_model_id
    # Try config.json for base_model
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            if "base_model" in config:
                return config["base_model"]
        except Exception:
            pass
    return None


def has_weights(model_dir):
    for fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]:
        if (model_dir / fname).exists():
            return True
    # Also check for subfolders with weights (e.g. for safetensors)
    for f in model_dir.iterdir():
        if f.is_dir() and any(x in f.name for x in ["weights", "checkpoints"]):
            if any((f / w).exists() for w in ["pytorch_model.bin", "model.safetensors", "tf_model.h5"]):
                return True
    return False


def download_weights(model_dir, base_model_id):
    print(f"  Downloading weights from base model: {base_model_id}")
    try:
        config = AutoConfig.from_pretrained(base_model_id)
        # Guess the right model class
        if config.architectures:
            arch = config.architectures[0].lower()
            if "causallm" in arch:
                model = AutoModelForCausalLM.from_pretrained(base_model_id)
            elif "seq2seq" in arch:
                model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
            elif "questionanswering" in arch:
                model = AutoModelForQuestionAnswering.from_pretrained(base_model_id)
            elif "tokenclassification" in arch:
                model = AutoModelForTokenClassification.from_pretrained(base_model_id)
            elif "speechseq2seq" in arch:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(base_model_id)
            else:
                model = AutoModel.from_pretrained(base_model_id)
        else:
            model = AutoModel.from_pretrained(base_model_id)
        model.save_pretrained(str(model_dir))
        print("  ✅ Weights downloaded and saved.")
        return True
    except Exception as e:
        print(f"  ❌ Failed to download weights: {e}")
        return False


def main():
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found.")
        return
    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith('.') or '_backup_' in model_dir.name:
            continue
        print(f"\nModel: {model_dir.name}")
        if has_weights(model_dir):
            print("  ✅ Weights found.")
            continue
        base_model_id = find_base_model_id(model_dir)
        if not base_model_id:
            print("  ❌ Could not determine base model identifier.")
            continue
        download_weights(model_dir, base_model_id)

if __name__ == "__main__":
    main()
