# Fine-tuning Mistral-7B for Pashto with AutoTrain

This guide explains how to fine-tune Mistral-7B language model on Pashto datasets using Hugging Face AutoTrain.

## Overview

[AutoTrain](https://huggingface.co/autotrain) is a Hugging Face service that allows you to easily fine-tune language models with minimal coding. This project uses AutoTrain to fine-tune the Mistral-7B model on Pashto datasets to create a specialized model for the ZamAI project.

## Prerequisites

1. A Hugging Face account with [Pro subscription](https://huggingface.co/pricing) (for access to AutoTrain)
2. Hugging Face API token with write access
3. Your dataset uploaded to the Hugging Face Hub in the correct format

## Dataset Format

Your dataset should be uploaded to the Hugging Face Hub and should have the following columns:

- `text`: For single-sequence training (complete context for training)
- OR
- `instruction`: For the prompt/instruction 
- `response`: For the expected model response

## Using the Script

The script `autotrain_finetune.py` handles the configuration and submission of your fine-tuning job to AutoTrain.

### Installation

Make sure you have the required packages installed:

```bash
pip install autotrain-advanced
```

### Running the Script

```bash
python autotrain_finetune.py \
  --project_name "ZamAI-Mistral-7B-Pashto" \
  --model "mistralai/Mistral-7B-v0.1" \
  --dataset "tasal9/ZamAI_Pashto_Training" \
  --text_column "text" \
  --lr 2e-4 \
  --epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --hf_token "YOUR_HF_TOKEN"
```

### Arguments

- `--project_name`: Name for your AutoTrain project
- `--model`: Base model to fine-tune (e.g., "mistralai/Mistral-7B-v0.1")
- `--dataset`: Dataset on HF Hub to use for training
- `--text_column`: Column name containing the formatted text
- `--lr`: Learning rate for training
- `--epochs`: Number of training epochs
- `--lora_r`: LoRA attention dimension 
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: LoRA attention dropout
- `--hf_token`: Hugging Face API token (required)

## Monitoring Your Training Job

After submitting your job:

1. Go to the [AutoTrain dashboard](https://ui.autotrain.huggingface.co/projects)
2. Find your project by name
3. Monitor training progress, logs, and metrics

## Using Your Fine-tuned Model

Once training is complete, your model will be automatically pushed to the Hugging Face Hub at:
`https://huggingface.co/tasal9/[PROJECT_NAME]`

You can then use the model like any other Hugging Face model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "tasal9/ZamAI-Mistral-7B-Pashto"  # Replace with your model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For LoRA models, you need to merge the adapter first
model = model.merge_and_unload()

# Generate text
prompt = "ستاسو نوم څه دی؟"  # What is your name?
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Additional Resources

- [Hugging Face AutoTrain Documentation](https://huggingface.co/docs/autotrain/index)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/index)
- [Mistral 7B Model](https://huggingface.co/mistralai/Mistral-7B-v0.1)
