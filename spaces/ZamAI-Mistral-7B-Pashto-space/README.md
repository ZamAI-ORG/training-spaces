---
title: ZamAI-Mistral-7B-Pashto Space
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu-a10g
---

# ZamAI-Mistral-7B-Pashto Space

This Hugging Face Space provides an interface for:

1. **Testing the ZamAI-Mistral-7B-Pashto model** - Try out text generation
2. **Fine-tuning the model** - Train on your own dataset
3. **Downloading your fine-tuned model** - Get your customized model

Uses ZeroGPU for efficient GPU acceleration.

## How to Use

### Test Model Tab
1. Enter your Pashto text in the input box
2. Click "Generate" to get the model's response
3. For best results, keep input under 512 characters

### Fine-tune Model Tab
1. Enter a Hugging Face dataset name (e.g., "username/dataset")
2. Set hyperparameters:
   - Learning rate (default: 5e-5)
   - Number of epochs (default: 3)
   - Batch size (default: 8)
3. Click "Start Fine-tuning"
4. Check status with "Check Status" button
5. Once complete, you can download your fine-tuned model

## Example Usage

**Input:**
```
سلام، څنګه یی؟
```

**Output:**
```
زه ښه یم، مننه!
```

## Training Data Format

The expected dataset format for fine-tuning is:
```json
{
  "train": [
    {"text": "Your training examples here"}
  ],
  "validation": [
    {"text": "Your validation examples here"}
  ]
}
```

You can also use the `instruction` and `response` format for instruction tuning:
```json
{
  "train": [
    {"instruction": "Your instruction", "response": "Expected response"}
  ]
}
```

---
