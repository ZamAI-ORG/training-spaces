#!/usr/bin/env python3
"""
Enhanced Space Template with Load Model Button and Advanced Features
"""

import gradio as gr
import spaces
import torch
import os
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from huggingface_hub import HfApi, upload_folder
import numpy as np

# Global variables to store model and tokenizer
MODEL = None
TOKENIZER = None
MODEL_LOADED = False
MODEL_LOADING_TIME = None

# Model configuration - replace with your model
MODEL_NAME = "MODEL_NAME_PLACEHOLDER"
MODEL_TYPE = "MODEL_TYPE_PLACEHOLDER"  # "causal_lm", "seq2seq", "text_classification", etc.

@spaces.GPU
def load_model():
    """Load the model and tokenizer with progress tracking"""
    global MODEL, TOKENIZER, MODEL_LOADED, MODEL_LOADING_TIME
    
    if MODEL_LOADED and MODEL is not None and TOKENIZER is not None:
        return "✅ Model already loaded and ready to use!"
    
    start_time = time.time()
    progress_updates = []
    
    try:
        progress_updates.append("🔍 Starting model loading process...")
        yield "\n".join(progress_updates)
        
        progress_updates.append("⏳ Loading tokenizer...")
        yield "\n".join(progress_updates)
        
        # Load tokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        if TOKENIZER.pad_token is None and TOKENIZER.eos_token is not None:
            TOKENIZER.pad_token = TOKENIZER.eos_token
        
        progress_updates.append("✅ Tokenizer loaded successfully")
        yield "\n".join(progress_updates)
        
        progress_updates.append(f"⏳ Loading model {MODEL_NAME} to GPU (this may take a while)...")
        yield "\n".join(progress_updates)
        
        # Load model with appropriate settings based on type
        if MODEL_TYPE == "causal_lm":
            MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # Default to causal language model if type not specified
            MODEL = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        MODEL_LOADED = True
        MODEL_LOADING_TIME = time.time() - start_time
        
        progress_updates.append(f"✅ Model loaded successfully in {MODEL_LOADING_TIME:.2f} seconds")
        progress_updates.append(f"🚀 Model is ready to use! You can now use the features below.")
        progress_updates.append(f"💡 RECOMMENDATION: Start by testing the model with a simple prompt to ensure it's working properly.")
        
        yield "\n".join(progress_updates)
        
    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"
        progress_updates.append(error_msg)
        yield "\n".join(progress_updates)
        MODEL_LOADED = False
        return "\n".join(progress_updates)

def check_model_loaded():
    """Check if model is loaded and return appropriate message"""
    if not MODEL_LOADED or MODEL is None or TOKENIZER is None:
        return False, "❌ Please load the model first using the 'Load Model' button at the top of the page."
    return True, "Model loaded and ready"

@spaces.GPU
def generate_text(input_text, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    """Generate text from the model"""
    # Check if model is loaded
    is_loaded, message = check_model_loaded()
    if not is_loaded:
        return message
    
    if not input_text.strip():
        return "Please enter a prompt to generate text."
    
    try:
        inputs = TOKENIZER(input_text, return_tensors="pt").to(MODEL.device)
        
        # Generate text with specified parameters
        with torch.no_grad():
            outputs = MODEL.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=TOKENIZER.eos_token_id
            )
        
        generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Return just the newly generated text without the prompt
        return generated_text[len(input_text):].strip()
        
    except Exception as e:
        return f"❌ Error during generation: {str(e)}"

def prepare_training_dataset(dataset_text):
    """Prepare training dataset from text input"""
    # Check if model is loaded
    is_loaded, message = check_model_loaded()
    if not is_loaded:
        return None, message
        
    lines = [line.strip() for line in dataset_text.split("\n") if line.strip()]
    
    if not lines:
        return None, "❌ Empty dataset. Please provide training examples."
    
    try:
        # Create a simple dataset
        dataset = Dataset.from_dict({"text": lines})
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return TOKENIZER(examples["text"], padding="max_length", truncation=True, max_length=512)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset, f"✅ Dataset prepared with {len(lines)} examples"
    
    except Exception as e:
        return None, f"❌ Failed to prepare dataset: {str(e)}"

@spaces.GPU
def train_model(dataset_text, epochs=1, learning_rate=2e-5, batch_size=2, save_model=False):
    """Train the model with actual implementation"""
    # Check if model is loaded
    is_loaded, message = check_model_loaded()
    if not is_loaded:
        return message
    
    if not dataset_text.strip():
        return "❌ Please provide training data."
    
    try:
        # Prepare dataset
        dataset, prep_message = prepare_training_dataset(dataset_text)
        if dataset is None:
            return prep_message
            
        progress_updates = []
        progress_updates.append(f"🔍 Starting training process...")
        progress_updates.append(f"📚 {prep_message}")
        yield "\n".join(progress_updates)
        
        # Training arguments
        output_dir = f"./results-{int(time.time())}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=float(learning_rate),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=MODEL,
            args=training_args,
            train_dataset=dataset,
        )
        
        progress_updates.append(f"🚀 Starting training for {epochs} epoch(s) with learning rate {learning_rate}...")
        yield "\n".join(progress_updates)
        
        # Train the model
        train_result = trainer.train()
        
        progress_updates.append(f"✅ Training complete!")
        progress_updates.append(f"📊 Training Loss: {train_result.training_loss:.4f}")
        progress_updates.append(f"⏱️ Training Time: {train_result.metrics['train_runtime']:.2f} seconds")
        
        # Save model if requested
        if save_model:
            model_save_dir = f"./trained-model-{int(time.time())}"
            trainer.save_model(model_save_dir)
            TOKENIZER.save_pretrained(model_save_dir)
            
            progress_updates.append(f"💾 Model saved to {model_save_dir}")
            progress_updates.append(f"📝 To use this model, you can upload it to the Hugging Face Hub using the 'Upload Model' tab.")
        
        progress_updates.append("\n💡 RECOMMENDATIONS AFTER TRAINING:")
        progress_updates.append("1. Test the model with new prompts to see how it performs")
        progress_updates.append("2. If results aren't satisfactory, try adjusting hyperparameters or training for more epochs")
        progress_updates.append("3. Consider increasing the dataset size for better results")
        
        yield "\n".join(progress_updates)
        
    except Exception as e:
        return f"❌ Training failed: {str(e)}"

@spaces.GPU
def evaluate_model(test_data, metric_choice="perplexity"):
    """Evaluate the model on test data"""
    # Check if model is loaded
    is_loaded, message = check_model_loaded()
    if not is_loaded:
        return message
    
    if not test_data.strip():
        return "❌ Please provide test data."
    
    try:
        # Split test data into examples
        test_examples = [example.strip() for example in test_data.split("\n") if example.strip()]
        
        results = []
        total_perplexity = 0
        
        for i, example in enumerate(test_examples):
            inputs = TOKENIZER(example, return_tensors="pt").to(MODEL.device)
            
            with torch.no_grad():
                outputs = MODEL(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = torch.exp(torch.tensor(loss)).item()
                
                total_perplexity += perplexity
                results.append(f"Example {i+1} - Perplexity: {perplexity:.4f}")
        
        avg_perplexity = total_perplexity / len(test_examples)
        
        final_result = "\n".join(results)
        final_result += f"\n\n📊 Average Perplexity: {avg_perplexity:.4f}"
        
        # Add recommendations
        final_result += "\n\n💡 RECOMMENDATIONS AFTER EVALUATION:"
        final_result += "\n1. Lower perplexity indicates better model performance"
        final_result += "\n2. If perplexity is high, consider additional training or fine-tuning"
        final_result += "\n3. Try comparing results across different model versions"
        
        return final_result
        
    except Exception as e:
        return f"❌ Evaluation failed: {str(e)}"

def upload_model_to_hub(model_dir, repo_name, token):
    """Upload trained model to HuggingFace Hub"""
    if not os.path.exists(model_dir):
        return "❌ Model directory not found. Please train a model first."
    
    if not repo_name.strip():
        return "❌ Please provide a repository name."
    
    if not token.strip():
        return "❌ Please provide your HuggingFace token."
    
    try:
        api = HfApi()
        
        # Create the repo if it doesn't exist
        try:
            api.create_repo(repo_id=repo_name, token=token, exist_ok=True)
        except Exception as e:
            return f"❌ Failed to create repository: {str(e)}"
        
        # Upload the model files
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_name,
            token=token,
            commit_message=f"Upload trained model from Spaces"
        )
        
        response = f"✅ Model successfully uploaded to {repo_name}!"
        response += "\n\n💡 RECOMMENDATIONS AFTER UPLOADING:"
        response += "\n1. You can now use this model in other applications by referencing its name"
        response += f"\n2. Try using it: `from transformers import AutoModel; model = AutoModel.from_pretrained('{repo_name}')`"
        response += "\n3. Share the model with others who might find it useful"
        
        return response
        
    except Exception as e:
        return f"❌ Upload failed: {str(e)}"

def model_info():
    """Display information about the loaded model"""
    if not MODEL_LOADED or MODEL is None:
        return "❌ Model not loaded. Please load the model first."
    
    info = f"# Model Information\n\n"
    info += f"- **Model Name**: {MODEL_NAME}\n"
    info += f"- **Model Type**: {MODEL_TYPE}\n"
    info += f"- **Loading Time**: {MODEL_LOADING_TIME:.2f} seconds\n\n"
    
    # Get model parameters
    total_params = sum(p.numel() for p in MODEL.parameters())
    trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    
    info += f"- **Total Parameters**: {total_params:,}\n"
    info += f"- **Trainable Parameters**: {trainable_params:,}\n"
    info += f"- **Model Device**: {next(MODEL.parameters()).device}\n\n"
    
    # Get tokenizer info
    vocab_size = len(TOKENIZER)
    info += f"- **Tokenizer Vocabulary Size**: {vocab_size:,}\n"
    info += f"- **Padding Token**: `{TOKENIZER.pad_token}`\n"
    info += f"- **EOS Token**: `{TOKENIZER.eos_token}`\n\n"
    
    info += "## Model Usage Recommendations\n\n"
    info += "1. **Testing**: Start with simple prompts to test the model's capabilities\n"
    info += "2. **Training**: Use domain-specific data for best results\n"
    info += "3. **Evaluation**: Regularly evaluate to track improvement\n"
    info += "4. **Parameters**: Experiment with temperature (0.7-1.0) for creative tasks, lower (0.2-0.5) for factual responses\n"
    
    return info

# Create Gradio interface
with gr.Blocks(title=f"{MODEL_NAME} Advanced Space", theme=gr.themes.Soft()) as iface:
    gr.Markdown(f"# {MODEL_NAME} Advanced Training Space")
    gr.Markdown("This space provides advanced functionality for training, testing, and using language models with ZeroGPU acceleration.")
    
    # Load model section - must be done first
    with gr.Box():
        gr.Markdown("### 🚀 Step 1: Load Model (Required)")
        with gr.Row():
            with gr.Column():
                load_btn = gr.Button("📥 Load Model", variant="primary", size="lg")
                gr.Markdown("⚠️ You must load the model before using any features below")
            with gr.Column():
                model_loading_output = gr.Markdown("Model not loaded. Click the button to load.")
        
        # Connect the load button
        load_btn.click(fn=load_model, outputs=model_loading_output)
    
    # Model Info Tab
    with gr.Accordion("ℹ️ Model Information", open=False):
        model_info_output = gr.Markdown("Load the model to see information")
        model_info_btn = gr.Button("📊 Show Model Information")
        model_info_btn.click(fn=model_info, outputs=model_info_output)
    
    # Main functionality tabs
    with gr.Tabs():
        # Test Tab
        with gr.TabItem("🧪 Test Model"):
            gr.Markdown("### Generate text with the model")
            with gr.Row():
                with gr.Column():
                    test_input = gr.Textbox(
                        label="Input Prompt",
                        placeholder="Enter text to test the model...",
                        lines=3
                    )
                    with gr.Row():
                        max_length_slider = gr.Slider(
                            minimum=10,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Max Output Length"
                        )
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.7,
                            label="Temperature"
                        )
                    with gr.Row():
                        top_p_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-p (nucleus sampling)"
                        )
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.2,
                            step=0.05,
                            label="Repetition Penalty"
                        )
                    test_btn = gr.Button("🚀 Generate", variant="primary")
                
                with gr.Column():
                    test_output = gr.Textbox(
                        label="Generated Output",
                        lines=8,
                        interactive=False
                    )
                    gr.Markdown("""
                    ### Parameter Guide
                    - **Temperature**: Higher values (>1) make output more random, lower values (<1) make it more focused and deterministic
                    - **Top-p**: Controls diversity by limiting tokens to the most probable ones that sum to probability p
                    - **Repetition Penalty**: Penalizes repetition of words/phrases (higher values reduce repetition)
                    """)
            
            test_btn.click(
                fn=generate_text,
                inputs=[test_input, max_length_slider, temperature_slider, top_p_slider, repetition_penalty_slider],
                outputs=test_output
            )
        
        # Train Tab
        with gr.TabItem("🏋️ Train Model"):
            gr.Markdown("### Train or fine-tune the model on your data")
            train_dataset = gr.Textbox(
                label="Training Dataset",
                placeholder="Enter training examples, one per line...",
                lines=8
            )
            with gr.Row():
                train_epochs = gr.Number(label="Epochs", value=1, minimum=1, maximum=10)
                train_lr = gr.Number(label="Learning Rate", value=2e-5, minimum=1e-6, maximum=1e-3)
                train_batch = gr.Number(label="Batch Size", value=2, minimum=1, maximum=8)
            
            train_save_model = gr.Checkbox(label="Save trained model locally", value=True)
            train_btn = gr.Button("🚀 Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Progress", lines=10, interactive=False)
            
            train_btn.click(
                fn=train_model,
                inputs=[train_dataset, train_epochs, train_lr, train_batch, train_save_model],
                outputs=train_output
            )
        
        # Evaluate Tab
        with gr.TabItem("📊 Evaluate Model"):
            gr.Markdown("### Evaluate model performance on test data")
            eval_dataset = gr.Textbox(
                label="Test Dataset",
                placeholder="Enter test examples, one per line...",
                lines=8
            )
            
            with gr.Row():
                metric_choice = gr.Radio(
                    ["perplexity", "accuracy"],
                    label="Evaluation Metric",
                    value="perplexity"
                )
            
            eval_btn = gr.Button("📊 Evaluate", variant="primary")
            eval_output = gr.Textbox(label="Evaluation Results", lines=8, interactive=False)
            
            eval_btn.click(
                fn=evaluate_model,
                inputs=[eval_dataset, metric_choice],
                outputs=eval_output
            )
        
        # Upload Tab
        with gr.TabItem("📤 Upload Model"):
            gr.Markdown("### Upload trained models to HuggingFace Hub")
            with gr.Row():
                model_dir_input = gr.Textbox(
                    label="Model Directory",
                    placeholder="./trained-model-1234567890",
                    lines=1
                )
                repo_name_input = gr.Textbox(
                    label="Repository Name",
                    placeholder="username/model-name",
                    lines=1
                )
            
            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                placeholder="hf_...",
                type="password",
                lines=1
            )
            
            upload_btn = gr.Button("📤 Upload to Hub", variant="primary")
            upload_output = gr.Textbox(label="Upload Status", lines=5, interactive=False)
            
            upload_btn.click(
                fn=upload_model_to_hub,
                inputs=[model_dir_input, repo_name_input, hf_token_input],
                outputs=upload_output
            )
    
    # Footer with recommendations
    gr.Markdown("""
    ## 💡 Recommendations for Working with this Model
    
    ### After Loading the Model:
    1. **Start by testing**: Use the Test tab with simple prompts to understand the model's capabilities
    2. **Evaluate baseline performance**: Run an evaluation on sample data before any training
    
    ### For Training:
    1. **Start small**: Begin with a small dataset and 1-2 epochs to test the training process
    2. **Use domain-specific data**: For best results, use data from your target domain
    3. **Monitor training loss**: If loss isn't decreasing, try adjusting the learning rate
    
    ### For Evaluation:
    1. **Use diverse test examples**: Include both simple and complex examples in your test set
    2. **Compare before/after**: Evaluate before and after training to measure improvement
    
    ### For Model Upload:
    1. **Use descriptive repo names**: Include model type and purpose in the repository name
    2. **Document your changes**: Add a good description when uploading your model
    
    ### General Tips:
    1. **Save checkpoints**: Always save your model after significant training
    2. **Track experiments**: Keep notes on hyperparameters and results
    3. **Start simple**: Master basic usage before attempting complex tasks
    """)

if __name__ == "__main__":
    iface.launch()
