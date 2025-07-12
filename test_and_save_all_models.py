
from huggingface_hub import model_info


# Check the first model folder: Multilingual-ZamAI-Embeddings
repo_name = "tasal9/Multilingual-ZamAI-Embeddings"
print(f"Checking Hugging Face Hub for model: {repo_name}")
try:
    info = model_info(repo_name)
    siblings = info.siblings if info.siblings is not None else []
    has_weights = any(
        f.rfilename.endswith(".bin") or f.rfilename.endswith(".safetensors")
        for f in siblings
    )
    if has_weights:
        print(f"Model weights found for {repo_name} on Hugging Face Hub.")
    else:
        print(f"No model weights found for {repo_name} on Hugging Face Hub.")
except Exception as e:
    print(f"Error checking {repo_name}: {e}")

SAMPLE_INPUT = "سلام، څنګه یی؟"

for repo_name, local_path in MODELS:
    print(f"\nProcessing {repo_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        model = AutoModelForCausalLM.from_pretrained(repo_name)
        print("Model loaded successfully.")
        # Test model
        inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Sample output: {result}")
        # Save model weights
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print(f"Model weights saved to {local_path}")
    except Exception as e:
        print(f"Error processing {repo_name}: {e}")
print("\nAll models processed.")
