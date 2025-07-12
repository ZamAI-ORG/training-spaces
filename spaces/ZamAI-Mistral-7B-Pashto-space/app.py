import gradio as gr
import spaces
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

# Model configuration
MODEL_NAME = "tasal9/ZamAI-Mistral-7B-Pashto"


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

# Create Gradio interface
with gr.Blocks(title="ZamAI-Mistral-7B-Pashto Space") as iface:
    gr.Markdown(f"# ZamAI-Mistral-7B-Pashto")
    gr.Markdown("""
    Example input: 
    > سلام، څنګه یی؟
    """)
    loading = gr.State(False)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input", lines=3, value="سلام، څنګه یی؟")
            submit_btn = gr.Button("Generate")
        with gr.Column():
            output_text = gr.Textbox(label="Output", lines=3)
    
    def wrapped_test_model(input_text):
        loading.set(True)
        result = test_model(input_text)
        loading.set(False)
        return result
    
    submit_btn.click(fn=test_model, inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    iface.launch()
