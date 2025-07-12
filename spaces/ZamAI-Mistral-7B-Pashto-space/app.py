import gradio as gr
import spaces
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import threading
from datetime import datetime

# Model configuration
MODEL_NAME = "tasal9/ZamAI-Mistral-7B-Pashto"

# Fine-tuning configuration
FINE_TUNING_STATUS = {
    "in_progress": False,
    "completed": False,
    "error": None,
    "progress": 0,
    "model_path": None
}

# Cache model and tokenizer
model_tokenizer_cache = {"model": None, "tokenizer": None, "loaded": False, "error": None}
model_lock = threading.Lock()

@spaces.GPU
def load_model():
    """Load the model and tokenizer, cache them"""
    with model_lock:
        if model_tokenizer_cache["loaded"]:
            return model_tokenizer_cache["model"], model_tokenizer_cache["tokenizer"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            model_tokenizer_cache["model"] = model
            model_tokenizer_cache["tokenizer"] = tokenizer
            model_tokenizer_cache["loaded"] = True
            model_tokenizer_cache["error"] = None
            return model, tokenizer
        except Exception as e:
            model_tokenizer_cache["error"] = str(e)
            return None, None

@spaces.GPU
def test_model(input_text):
    """Test the model with given input"""
    # Input validation
    if not isinstance(input_text, str) or len(input_text.strip()) == 0:
        return "Please enter some text to generate."
    if len(input_text) > 512:
        return "Input too long (max 512 characters)."

    model, tokenizer = load_model()
    if model is None or tokenizer is None:
        error_msg = model_tokenizer_cache["error"] or "Failed to load model."
        return f"Model loading error: {error_msg}"
    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Model inference error: {str(e)}"

@spaces.GPU
def finetune_model(dataset_name, learning_rate, num_epochs, batch_size, progress=gr.Progress()):
    """Fine-tune the model on a given dataset"""
    FINE_TUNING_STATUS["in_progress"] = True
    FINE_TUNING_STATUS["completed"] = False
    FINE_TUNING_STATUS["error"] = None
    FINE_TUNING_STATUS["progress"] = 0
    
    try:
        # Load dataset
        progress(0.1, desc="Loading dataset...")
        dataset = load_dataset(dataset_name)
        progress(0.2, desc="Dataset loaded")
        
        # Load model and tokenizer
        progress(0.3, desc="Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        progress(0.4, desc="Model and tokenizer loaded")
        
        # Prepare dataset
        progress(0.5, desc="Preparing dataset...")
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        progress(0.6, desc="Dataset prepared")
        
        # Define training arguments
        output_dir = f"fine-tuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(learning_rate),
            num_train_epochs=int(num_epochs),
            per_device_train_batch_size=int(batch_size),
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=2,
        )
        
        # Create trainer
        progress(0.7, desc="Setting up trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer,
        )
        
        # Train model
        progress(0.8, desc="Training model...")
        trainer.train()
        progress(0.9, desc="Training complete")
        
        # Save model
        progress(0.95, desc="Saving model...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        FINE_TUNING_STATUS["completed"] = True
        FINE_TUNING_STATUS["in_progress"] = False
        FINE_TUNING_STATUS["progress"] = 100
        FINE_TUNING_STATUS["model_path"] = output_dir
        
        progress(1.0, desc="Fine-tuning complete!")
        return f"Fine-tuning completed successfully. Model saved to {output_dir}"
    except Exception as e:
        FINE_TUNING_STATUS["error"] = str(e)
        FINE_TUNING_STATUS["in_progress"] = False
        return f"Fine-tuning failed: {str(e)}"

def get_finetune_status():
    """Get the current status of fine-tuning"""
    if FINE_TUNING_STATUS["in_progress"]:
        return f"Fine-tuning in progress... ({FINE_TUNING_STATUS['progress']}%)"
    elif FINE_TUNING_STATUS["completed"]:
        return f"Fine-tuning completed. Model saved to {FINE_TUNING_STATUS['model_path']}"
    elif FINE_TUNING_STATUS["error"]:
        return f"Fine-tuning failed: {FINE_TUNING_STATUS['error']}"
    else:
        return "No fine-tuning has been started yet."

# Create Gradio interface
with gr.Blocks(title="ZamAI-Mistral-7B-Pashto Space") as iface:
    gr.Markdown(f"# ZamAI-Mistral-7B-Pashto")
    
    with gr.Tabs():
        with gr.TabItem("Test Model"):
            gr.Markdown("""
            Test the ZamAI-Mistral-7B-Pashto model with your own text.
            
            Example input: 
            > سلام، څنګه یی؟
            """)
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Input", lines=3, value="سلام، څنګه یی؟")
                    submit_btn = gr.Button("Generate")
                with gr.Column():
                    output_text = gr.Textbox(label="Output", lines=3)
            
            submit_btn.click(fn=test_model, inputs=input_text, outputs=output_text)
            
        with gr.TabItem("Fine-tune Model"):
            gr.Markdown("""
            Fine-tune the model on your own dataset. 
            
            The dataset should be available on Hugging Face Hub and contain a 'text' column.
            """)
            
            dataset_name = gr.Textbox(label="Dataset Name (e.g., 'username/dataset')", value="")
            with gr.Row():
                learning_rate = gr.Number(label="Learning Rate", value=5e-5)
                num_epochs = gr.Number(label="Number of Epochs", value=3, precision=0)
                batch_size = gr.Number(label="Batch Size", value=8, precision=0)
            
            finetune_btn = gr.Button("Start Fine-tuning")
            finetune_output = gr.Textbox(label="Fine-tuning Status", interactive=False)
            
            finetune_btn.click(
                fn=finetune_model, 
                inputs=[dataset_name, learning_rate, num_epochs, batch_size],
                outputs=finetune_output
            )
            
            status_btn = gr.Button("Check Status")
            status_btn.click(fn=get_finetune_status, inputs=None, outputs=finetune_output)

if __name__ == "__main__":
    iface.launch()
