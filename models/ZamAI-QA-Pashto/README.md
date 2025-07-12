---
license: apache-2.0
language:
- ps
- en
library_name: transformers
pipeline_tag: question-answering
tags:
- pashto
- afghanistan
- zamai
- conversational-ai
- instruction-tuning
datasets:
- tasal9/ZamAI_Pashto_Dataset
metrics:
- perplexity
- bleu
widget:
- text: "سلام دې وي! تاسو څنګه یاست؟"
  example_title: "Pashto Greeting"
- text: "د افغانستان د تاریخ په اړه راته ووایه"
  example_title: "Afghanistan History"
- text: "Hello, how can I help you today?"
  example_title: "English Greeting"
---

# ZamAI-QA-Pashto

## Model Description

Question-answering model specialized for Pashto knowledge queries and factual information retrieval.

This model is part of the ZamAI (زمای) project - an advanced Afghan AI assistant designed to understand and communicate in Pashto, English, and other Afghan languages.

## Key Features

- Factual question answering
- Afghan cultural knowledge
- Historical information
- Educational content QA
- Context-aware responses

## Use Cases

- Educational assistance
- Research support
- Information retrieval
- Study companion
- Knowledge base queries

## Model Architecture

- **Base Model:** microsoft/DialoGPT-medium
- **Architecture:** gpt2
- **Task:** question-answering
- **Languages:** Pashto (ps), English (en)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "tasal9/zamai-qa-pashto"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate text
prompt = "سلام! زه د افغانستان په اړه پوښتنه لرم:"
inputs = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

- **Dataset:** ZamAI Pashto Dataset (tasal9/ZamAI_Pashto_Dataset)
- **Training Method:** Fine-tuning on QA datasets
- **Epochs:** 4
- **Batch Size:** 3
- **Learning Rate:** 5e-5

## Performance

The model has been trained on conversational Pashto data and shows strong performance in:
- Natural conversation flow
- Cultural context understanding
- Mixed language handling (Code-switching)
- Afghan cultural knowledge

## Limitations

- Primary focus on Pashto and English
- May require further fine-tuning for specific domains
- Performance may vary with complex technical terminology

## Ethical Considerations

This model is designed to respect Afghan and Islamic values, promoting positive and constructive conversations while avoiding harmful or inappropriate content.

## Citation

```bibtex
@misc{zamai_zamai_qa_pashto_2024,
  title={ZamAI ZamAI-QA-Pashto: Advanced Pashto Language Model},
  author={ZamAI Team},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/tasal9/zamai-qa-pashto}
}
```

## Contact

For questions, suggestions, or collaboration opportunities, please reach out through the ZamAI project.

---

*Built with ❤️ for the Afghan community*
