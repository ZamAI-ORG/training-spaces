import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
MODEL_NAME = "tasal9/zamai-dialogpt-pashto-v3"

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
def test_model(input_text):
    """Test the model with given input"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return "Failed to load model"
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="zamai-dialogpt-pashto-v3 Space") as iface:
    gr.Markdown(f"# zamai-dialogpt-pashto-v3")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input", lines=3)
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            output_text = gr.Textbox(label="Output", lines=3)
    
    submit_btn.click(fn=test_model, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    iface.launch()
