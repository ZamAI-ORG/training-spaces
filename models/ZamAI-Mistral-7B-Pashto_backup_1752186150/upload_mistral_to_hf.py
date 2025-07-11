#!/usr/bin/env python3
"""
ZamAI Mistral Model Upload Script
This script uploads the ZamAI-Mistral-7B-Pashto model repository to Hugging Face Hub
"""

import os
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def create_mistral_model_card():
    """Create a comprehensive model card for the ZamAI Mistral model"""
    model_card_content = """---
license: apache-2.0
language:
  - ps
  - fa
  - ar
  - en
base_model:
  - mistralai/Mistral-7B-v0.1
pipeline_tag: text-generation
library_name: transformers
tags:
  - mistral
  - pashto
  - afghanistan
  - fine-tuned
  - zamai
  - multilingual
  - text-generation
  - lora
  - autotrain
widget:
  - text: "ستاسو نوم څه دی؟"
    example_title: "Pashto Question"
  - text: "د افغانستان د کلتور په اړه ماته معلومات راکړئ."
    example_title: "Cultural Question"
  - text: "What is the capital of Afghanistan?"
    example_title: "English Question"
model-index:
  - name: ZamAI-Mistral-7B-Pashto
    results: []
---

# ZamAI-Mistral-7B-Pashto

This is a fine-tuned version of Mistral-7B-v0.1 specifically optimized for Pashto language understanding and generation. It's part of the ZamAI project, focusing on advancing AI capabilities for Afghanistan's primary languages.

## Model Information

- **Base Model**: [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- **Fine-tuned On**: Pashto language data ([tasal9/ZamAI_Pashto_Dataset](https://huggingface.co/datasets/tasal9/ZamAI_Pashto_Dataset))
- **Training Method**: AutoTrain with LoRA fine-tuning
- **Languages**: Primarily Pashto, with cross-lingual capabilities
- **Task**: Text Generation, Question Answering, Instruction Following

## Features

- **Pashto Language Expertise**: Trained specifically on Pashto text data
- **Cultural Context**: Understanding of Afghan and Pashtun cultural contexts
- **Instruction Following**: Capable of following instructions in Pashto
- **Code-Switching**: Handles mixed Pashto-English conversations
- **Text Generation**: Generates coherent Pashto text

## Usage

### Using Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "tasal9/ZamAI-Mistral-7B-Pashto"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
prompt = "ستاسو نوم څه دی؟"  # What is your name?
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with Pipeline

```python
from transformers import pipeline

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model="tasal9/ZamAI-Mistral-7B-Pashto",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
result = generator(
    "د افغانستان د کلتور په اړه ماته معلومات راکړئ.",
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

print(result[0]['generated_text'])
```

## Training Details

### Training Data
- **Dataset**: Custom Pashto language dataset
- **Size**: Multiple thousand examples of Pashto text
- **Format**: Instruction-response pairs and conversational data
- **Quality**: Manually curated and filtered for accuracy

### Training Configuration
- **Method**: LoRA (Low-Rank Adaptation) fine-tuning
- **Framework**: Hugging Face AutoTrain
- **Parameters**: 
  - Learning Rate: 2e-4
  - Epochs: 3-5
  - LoRA Rank: 16-32
  - LoRA Alpha: 32-64
  - Batch Size: 4

### Hardware
- **Platform**: Hugging Face AutoTrain infrastructure
- **GPU**: High-end NVIDIA GPUs with sufficient VRAM

## Evaluation

The model has been evaluated on various Pashto language tasks:

- **Text Generation Quality**: High coherence in Pashto responses
- **Instruction Following**: Good performance on Pashto instructions
- **Cultural Accuracy**: Contextually appropriate responses
- **Language Consistency**: Maintains Pashto throughout conversations

## Limitations

- **Language Scope**: Primarily optimized for Pashto; other languages may have reduced performance
- **Cultural Context**: Trained primarily on available Pashto data; may not cover all regional variations
- **Technical Terms**: May struggle with highly technical or specialized vocabulary
- **Safety**: Please use responsibly and be aware of potential biases in generated content

## Intended Use

This model is intended for:
- Educational applications in Pashto language
- Content generation for Pashto speakers
- Research in multilingual AI systems
- Building AI applications for Afghanistan and Pashtun communities

## Ethical Considerations

- The model should be used responsibly and ethically
- Generated content should be reviewed for accuracy and appropriateness
- Users should be aware of potential biases in AI-generated content
- Respect cultural sensitivities when using the model

## Technical Details

### Model Architecture
- **Type**: Causal Language Model (Decoder-only Transformer)
- **Parameters**: ~7B (base Mistral-7B)
- **Fine-tuning**: LoRA adapters for efficient training
- **Context Length**: 32k tokens (inherited from Mistral-7B)

### Files and Formats
- **Model Weights**: Available in Hugging Face format
- **Tokenizer**: Optimized for Pashto text processing
- **Configuration**: Standard Mistral configuration with adaptations

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{zamai-mistral-7b-pashto,
  title={ZamAI-Mistral-7B-Pashto: A Fine-tuned Language Model for Pashto},
  author={ZamAI Team},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto}
}
```

## License

This model is released under the Apache 2.0 license, following the base Mistral-7B model license.

## Contact

For questions, issues, or collaborations:
- **GitHub**: [ZamAI Project](https://github.com/tasal9/Huggingface)
- **Hugging Face**: [tasal9](https://huggingface.co/tasal9)

## Acknowledgments

- **Mistral AI** for the excellent base Mistral-7B model
- **Hugging Face** for AutoTrain and the transformers library
- **The Pashto Language Community** for cultural and linguistic insights
- **Contributors** to the ZamAI project

---

*Built with ❤️ for the Afghan and Pashtun communities worldwide.*
"""
    
    return model_card_content

def upload_mistral_to_hf():
    """Upload the ZamAI Mistral model to Hugging Face Hub"""
    
    # Read the HF token
    hf_token_path = "../Multilingual-ZamAI-Embeddings/HF-Token.txt"
    if os.path.exists(hf_token_path):
        with open(hf_token_path, 'r') as f:
            token = f.read().strip()
    else:
        print("❌ HF Token file not found. Please ensure HF-Token.txt exists.")
        return False
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Repository details
    repo_id = "tasal9/ZamAI-Mistral-7B-Pashto"
    repo_type = "model"
    
    print(f"🚀 Uploading ZamAI Mistral-7B-Pashto to {repo_id}")
    
    try:
        # Create repository if it doesn't exist
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            exist_ok=True,
            private=False
        )
        print(f"✅ Repository {repo_id} is ready")
        
        # Create model card
        model_card = create_mistral_model_card()
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(model_card)
        print("✅ Model card created")
        
        # Upload the entire folder
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message="Upload ZamAI-Mistral-7B-Pashto model repository",
            ignore_patterns=[".git/", "__pycache__/", "*.pyc", ".gitignore"]
        )
        
        print(f"🎉 Successfully uploaded to https://huggingface.co/{repo_id}")
        print("Your Mistral model repository is now live on Hugging Face Hub!")
        
        # Additional information
        print("\n📚 Repository Contents:")
        print("  - Fine-tuning scripts and workflows")
        print("  - Evaluation and iteration tools")
        print("  - Comprehensive documentation")
        print("  - Usage examples and guides")
        
        print(f"\n🔗 Direct Links:")
        print(f"  - Model: https://huggingface.co/{repo_id}")
        print(f"  - Files: https://huggingface.co/{repo_id}/tree/main")
        print(f"  - Usage: https://huggingface.co/{repo_id}#usage")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = upload_mistral_to_hf()
    if success:
        print("\n✨ ZamAI-Mistral-7B-Pashto deployment completed successfully!")
    else:
        print("\n❌ Deployment failed. Please check the errors above.")
