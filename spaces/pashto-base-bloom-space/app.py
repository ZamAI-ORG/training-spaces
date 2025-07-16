import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import os

# Model configuration
MODEL_NAME = "tasal9/pashto-base-bloom"

@spaces.GPU
def load_model():
    """Load the model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return model, tokenizer
    except Exception as e:
        return None, None

@spaces.GPU
def test_model(input_text, max_length, temperature, top_p):
    """Test the model with given input"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "Failed to load model"
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU
def train_model(dataset_path, epochs, learning_rate, status=gr.Progress()):
    """Train the model on the given dataset"""
    try:
        status(0, desc="Loading dataset...")
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        status(0.2, desc="Loading model...")
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "Failed to load model"
        status(0.3, desc="Preparing training arguments...")
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=2,
            logging_dir="./logs",
            save_strategy="epoch"
        )
        status(0.4, desc="Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        status(0.5, desc="Training...")
        trainer.train()
        status(0.9, desc="Saving model...")
        model.save_pretrained("./trained_model")
        tokenizer.save_pretrained("./trained_model")
        status(1, desc="Done!")
        return "Training complete. Model saved to ./trained_model"
    except Exception as e:
        return f"Error: {str(e)}"

@spaces.GPU
def finetune_model(dataset_path, epochs, learning_rate, status=gr.Progress()):
    """Fine-tune the model on the given dataset (demo version)"""
    # For demo, same as train_model, but could load a checkpoint
    return train_model(dataset_path, epochs, learning_rate, status)

# Create Gradio interface
with gr.Blocks(title="pashto-base-bloom Space") as iface:
    gr.Markdown("# pashto-base-bloom\nTest, Train, and Fine-tune your Pashto model!")
    with gr.Tab("Test Model"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(label="Input", lines=3)
                max_length = gr.Slider(label="Max Length", minimum=10, maximum=256, value=100)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=1.0)
                top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, value=0.95)
                submit_btn = gr.Button("Generate")
            with gr.Column():
                output_text = gr.Textbox(label="Output", lines=6)
        submit_btn.click(fn=test_model, inputs=[input_text, max_length, temperature, top_p], outputs=output_text)
    with gr.Tab("Train Model"):
        gr.Markdown("Upload a JSON dataset for training. Format: [{'text': ...}]")
        dataset_file = gr.File(label="Training Dataset (JSON)")
        epochs = gr.Number(label="Epochs", value=3, minimum=1, maximum=10)
        learning_rate = gr.Number(label="Learning Rate", value=5e-5)
        train_btn = gr.Button("Start Training")
        train_status = gr.Textbox(label="Training Status", lines=6)
        train_btn.click(fn=train_model, inputs=[dataset_file, epochs, learning_rate], outputs=train_status)
    with gr.Tab("Fine-tune Model"):
        gr.Markdown("Upload a JSON dataset for fine-tuning. Format: [{'text': ...}]")
        finetune_file = gr.File(label="Fine-tuning Dataset (JSON)")
        ft_epochs = gr.Number(label="Epochs", value=3, minimum=1, maximum=10)
        ft_learning_rate = gr.Number(label="Learning Rate", value=5e-5)
        finetune_btn = gr.Button("Start Fine-tuning")
        finetune_status = gr.Textbox(label="Fine-tuning Status", lines=6)
        finetune_btn.click(fn=finetune_model, inputs=[finetune_file, ft_epochs, ft_learning_rate], outputs=finetune_status)

if __name__ == "__main__":
    iface.launch()
