---
language:
- en
- ps
license: mit
library_name: transformers
tags:
- phi-3
- business
- document-processing
- form-automation
- pashto
base_model: microsoft/Phi-3-mini-4k-instruct
pipeline_tag: text-generation
datasets:
- tasal9/Pashto-Dataset-Creating-Dataset
---

# ZamAI-Phi-3-Mini-Pashto

Phi-3-mini fine-tuned for business document processing and form automation

## 🌟 Model Overview

This model is part of the **ZamAI Pro Models Strategy** - a comprehensive AI platform designed for multilingual applications with specialized focus on Pashto language support.

### Key Features
- 🧠 **Advanced AI**: Based on microsoft/Phi-3-mini-4k-instruct architecture
- 🌐 **Multilingual**: Optimized for Pashto and English
- ⚡ **High Performance**: Optimized for production deployment
- 🔒 **Secure**: Enterprise-grade security and privacy

## 📚 Usage

### Basic Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto")
model = AutoModel.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto")

# Example usage
text = "Your input text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

### Usage with Hugging Face Inference API

```python
from huggingface_hub import InferenceClient

client = InferenceClient(token="your_hf_token")

response = client.text_generation(
    model="tasal9/ZamAI-Phi-3-Mini-Pashto",
    prompt="Your prompt here",
    max_new_tokens=200
)
```

## 🔧 Technical Details

- **Model Type**: text-generation
- **Base Model**: microsoft/Phi-3-mini-4k-instruct
- **Languages**: Pashto (ps), English (en)
- **License**: MIT
- **Training**: Fine-tuned on Pashto educational and cultural content

## 🚀 Applications

This model powers:
- **ZamAI Educational Platform**: Pashto language tutoring
- **Business Automation**: Document processing and analysis  
- **Voice Assistants**: Natural language understanding
- **Cultural Preservation**: Supporting Pashto language technology

## 📞 Support

For support and integration assistance:
- 📧 **Email**: support@zamai.ai
- 🌐 **Website**: [zamai.ai](https://zamai.ai)
- 💬 **Community**: [ZamAI Community](https://community.zamai.ai)

## 📄 License

Licensed under the MIT License.

---

**Part of the ZamAI Pro Models Strategy - Transforming AI for Multilingual Applications** 🌟

*Updated: 2025-07-05 21:29:11 UTC*
