# ZamAI Pashto LLM Fine-tuning Workflow

This guide provides a comprehensive workflow for fine-tuning a Mistral-7B# After training completes, your model will be available at `tasal9/ZamAI-Mistral-7B-Pashto`. Use it in your applications:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "tasal9/ZamAI-Mistral-7B-Pashto"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) Pashto data using Hugging Face AutoTrain.

## Overview of the Workflow

1. **Prepare the Dataset**: Standardize column names and split into train/test
2. **Check Dataset Format**: Validate dataset compatibility with AutoTrain
3. **Upload to Hugging Face**: Share dataset on the Hub for use with AutoTrain
4. **Start Fine-tuning**: Launch a training job with AutoTrain
5. **Evaluate Results**: Check model performance and iterate if needed

## Step 1: Prepare Your Data

Use the `prepare_for_autotrain.py` script to format your data for AutoTrain:

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install datasets huggingface_hub tqdm

# Format data - choose either "text" or "instruction-response" format
python prepare_for_autotrain.py \
  --input_file zamai_pashto_dataset.json \
  --format_type instruction-response \
  --push_to_hub \
  --hub_dataset_name ZamAI_Pashto_Training \
  --hf_token YOUR_HF_TOKEN_HERE
```

This script:
- Converts your data to a standardized format
- Splits into train/test sets
- Uploads to the Hugging Face Hub

## Step 2: Verify Dataset Format

Use the `check_dataset_format.py` script to validate that your dataset is ready for training:

```bash
# Check the uploaded dataset format
python check_dataset_format.py --dataset_path YOUR_USERNAME/ZamAI_Pashto_Training
```

Ensure that:
- The dataset has the expected columns (`text` or `instruction`/`response`)
- Both train and test splits are available

## Step 3: Fine-tune with AutoTrain

Use the `autotrain_finetune.py` script to launch a fine-tuning job on AutoTrain:

```bash
python autotrain_finetune.py \
  --project_name "ZamAI-Mistral-7B-Pashto" \
  --model "mistralai/Mistral-7B-v0.1" \
  --dataset "tasal9/ZamAI_Pashto_Dataset" \
  --text_column "instruction" \
  --lr 2e-4 \
  --epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --hf_token "YOUR_HF_TOKEN_HERE"
```

## Step 4: Monitor Training Progress

Visit the [AutoTrain dashboard](https://ui.autotrain.huggingface.co/projects) to monitor:
- Training logs
- Loss and evaluation metrics
- Resource utilization

## Step 5: Evaluate and Iterate

After training, evaluate your model's performance using our evaluation script:

```bash
# Evaluate your fine-tuned model
python evaluate_and_iterate.py \
  --model_name "tasal9/ZamAI-Mistral-7B-Pashto" \
  --dataset_file "zamai_pashto_dataset.json" \
  --num_samples 20 \
  --output_file "evaluation_results.json"
```

The script will:
1. **Select diverse samples** from different categories in your dataset
2. **Generate responses** using your fine-tuned model
3. **Analyze performance** and provide recommendations for improvement
4. **Save detailed results** for further analysis

Based on the evaluation:

1. **Identify areas for improvement** in your dataset or training setup
2. **Update your dataset** by adding more examples where needed
3. **Iterate by adjusting**:
   - Dataset composition and cleaning
   - Model hyperparameters (learning rate, epochs, etc.)
   - LoRA parameters for more efficient fine-tuning

### Parameters to Experiment With

| Parameter | Range to Try | Impact |
|-----------|--------------|--------|
| Learning Rate | 1e-5 to 5e-4 | Lower values are more stable but slower |
| Epochs | 2-5 | More epochs may improve performance but risk overfitting |
| LoRA rank (r) | 8-64 | Higher ranks give more capacity but use more memory |
| LoRA alpha | 16-64 | Controls adaptation strength |

## Using Your Fine-tuned Model

After training completes, your model will be available at `YOUR_USERNAME/ZamAI-Mistral-7B-Pashto`. Use it in your applications:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "YOUR_USERNAME/ZamAI-Mistral-7B-Pashto"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For LoRA models, you need to merge the adapter (optional)
# from peft import PeftModel
# model = PeftModel.from_pretrained(model, model_name)
# model = model.merge_and_unload()

# Generate text
prompt = "ستاسو نوم څه دی؟"  # What is your name?
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Troubleshooting

If you encounter issues:

1. **Dataset formatting problems**: Check column names and dataset splits
2. **Out-of-memory errors**: Reduce batch size or use LoRA with smaller rank
3. **Poor performance**: Try cleaning your dataset, adding more examples, or using different training parameters

## Pushing Files to GitHub

After fine-tuning, you'll want to organize and push your model files to GitHub. Use our automated scripts:

```bash
# Update the GitHub repository with fine-tuning files
./update_github.sh
```

This script will:

1. Create the proper directory structure in `models/text-generation/ZamAI-Mistral-7B/`
2. Copy all relevant files (scripts, documentation, configs) to this directory
3. Commit and push changes to GitHub

You can also use the combined update script to update both GitHub and Hugging Face Hub:

```bash
# Update both GitHub and Hugging Face Hub
./update_repos.sh
```

### Manual Organization (if needed)

If you prefer to organize files manually:

1. Create the model directory if it doesn't exist:

   ```bash
   mkdir -p models/text-generation/ZamAI-Mistral-7B
   ```

2. Copy relevant files to the model directory:

   ```bash
   cp autotrain_finetune.py evaluate_and_iterate.py FINETUNING_WORKFLOW.md models/text-generation/ZamAI-Mistral-7B/
   ```

3. Add a model-specific README:

   ```bash
   cp README.md models/text-generation/ZamAI-Mistral-7B/MODEL_README.md
   ```

4. Commit and push to GitHub:

   ```bash
   git add models/
   git commit -m "Add ZamAI-Mistral-7B fine-tuning files"
   git push origin main
   ```

## Resources

- [AutoTrain Documentation](https://huggingface.co/docs/autotrain/index)
- [PEFT Library Documentation](https://huggingface.co/docs/peft/index)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/main/en/training)
