import gradio as gr
import time
import threading
import random
from datetime import datetime

# Global state to track training/fine-tuning status
class TrainingState:
    def __init__(self):
        self.status = "idle"
        self.progress = 0
        self.logs = []
        self.start_time = None
        self.model_name = "tasal9/pashto-base-bloom"
        self.active_process = None

    def start_training(self, data_size):
        self.status = "training"
        self.progress = 0
        self.logs = [f"Training started at {datetime.now().strftime('%H:%M:%S')}"]
        self.logs.append(f"Training data size: {data_size} characters")
        self.start_time = time.time()
        
    def start_finetuning(self, data_size):
        self.status = "fine-tuning"
        self.progress = 0
        self.logs = [f"Fine-tuning started at {datetime.now().strftime('%H:%M:%S')}"]
        self.logs.append(f"Fine-tuning data size: {data_size} characters")
        self.start_time = time.time()
        
    def update_progress(self, progress):
        self.progress = min(100, max(0, progress))
        if progress >= 100 and self.status != "idle":
            self.complete_process()
            
    def add_log(self, message):
        self.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        if len(self.logs) > 10:  # Keep only last 10 logs
            self.logs.pop(0)
            
    def complete_process(self):
        elapsed = time.time() - self.start_time
        self.add_log(f"{self.status.capitalize()} completed in {elapsed:.1f} seconds!")
        self.status = "idle"
        self.progress = 100
        
    def get_status(self):
        status_map = {
            "idle": "✅ Ready",
            "training": "🔄 Training in progress",
            "fine-tuning": "🔄 Fine-tuning in progress"
        }
        return status_map.get(self.status, "❓ Unknown status")

# Create global state
state = TrainingState()

def test_model(input_text):
    """Enhanced test function with response variations"""
    if not input_text.strip():
        return "Please enter some text to test."
    
    responses = [
        f"Processed: '{input_text}'",
        f"Model response to: {input_text}",
        f"Analysis: This appears to be Pashto text with {len(input_text)} characters",
        f"✅ Received: {input_text}",
        f"Generated continuation: {input_text}... [simulated output]"
    ]
    return random.choice(responses)

def simulate_process(duration, process_type, data_size):
    """Simulate long-running training/fine-tuning process"""
    if process_type == "train":
        state.start_training(data_size)
    else:
        state.start_finetuning(data_size)
    
    steps = 10
    for i in range(steps + 1):
        time.sleep(duration / steps)
        progress = int((i / steps) * 100)
        state.update_progress(progress)
        
        # Add simulated log messages
        if i % 3 == 0:
            messages = [
                f"Processing batch {i*5}/{steps*5}",
                f"Loss: {random.uniform(0.1, 1.0):.4f}",
                f"Accuracy: {random.uniform(80, 95):.1f}%",
                f"Learning rate: {random.uniform(1e-5, 1e-3):.6f}"
            ]
            state.add_log(random.choice(messages))
    
    state.complete_process()

def train_model(dataset_text):
    """Training function with simulated processing"""
    if not dataset_text.strip():
        return "Please provide training data.", ""
    
    data_size = len(dataset_text)
    if state.status != "idle":
        return "Another process is already running. Please wait.", ""
    
    # Start simulation in background thread
    threading.Thread(
        target=simulate_process,
        args=(15, "train", data_size),
        daemon=True
    ).start()
    
    return "Training started successfully! Check status in the Status tab.", ""

def finetune_model(dataset_text):
    """Fine-tuning function with simulated processing"""
    if not dataset_text.strip():
        return "Please provide fine-tuning data.", ""
    
    data_size = len(dataset_text)
    if state.status != "idle":
        return "Another process is already running. Please wait.", ""
    
    # Start simulation in background thread
    threading.Thread(
        target=simulate_process,
        args=(10, "fine-tune", data_size),
        daemon=True
    ).start()
    
    return "Fine-tuning started successfully! Check status in the Status tab.", ""

def get_current_status():
    """Get current system status"""
    status_text = state.get_status()
    
    # Add progress information
    if state.status != "idle":
        status_text += f" - {state.progress}% complete"
    
    # Format logs
    logs = "\n".join(state.logs) if state.logs else "No logs available"
    
    return {
        status_box: status_text,
        progress_bar: state.progress / 100,
        log_output: logs
    }

# Create interface
with gr.Blocks(title="Pashto-Base-Bloom Trainer", theme="soft") as demo:
    gr.Markdown("# 🌸 Pashto-Base-Bloom Training Space")
    gr.Markdown("Train and fine-tune Pashto language model tasal9/pashto-base-bloom")
    
    with gr.Tab("Test Model"):
        gr.Markdown("### Test Model with Sample Text")
        with gr.Row():
            with gr.Column():
                test_input = gr.Textbox(label="Input Text", lines=3, placeholder="Enter Pashto text here...")
                test_btn = gr.Button("Run Test", variant="primary")
            test_output = gr.Textbox(label="Model Output", lines=4, interactive=False)
        test_btn.click(test_model, inputs=test_input, outputs=test_output)
    
    with gr.Tab("Train Model"):
        gr.Markdown("### Train Model with New Data")
        with gr.Row():
            with gr.Column():
                train_input = gr.Textbox(label="Training Data", lines=8, placeholder="Paste training dataset here...")
                train_btn = gr.Button("Start Training", variant="primary")
            train_output = gr.Textbox(label="Training Status", lines=2, interactive=False)
        train_btn.click(train_model, inputs=train_input, outputs=train_output)
    
    with gr.Tab("Fine-tune Model"):
        gr.Markdown("### Fine-tune Model with Specialized Data")
        with gr.Row():
            with gr.Column():
                finetune_input = gr.Textbox(label="Fine-tuning Data", lines=8, placeholder="Paste fine-tuning dataset here...")
                finetune_btn = gr.Button("Start Fine-tuning", variant="primary")
            finetune_output = gr.Textbox(label="Fine-tuning Status", lines=2, interactive=False)
        finetune_btn.click(finetune_model, inputs=finetune_input, outputs=finetune_output)
    
    with gr.Tab("Status"):
        gr.Markdown("### System Status")
        with gr.Row():
            with gr.Column():
                status_box = gr.Textbox(label="Current Status", interactive=False)
                progress_bar = gr.ProgressBar(label="Progress")
                refresh_btn = gr.Button("Refresh Status", variant="secondary")
                auto_refresh = gr.Checkbox(label="Auto-refresh every 5 seconds", value=True)
            log_output = gr.Textbox(label="Process Logs", lines=8, interactive=False)
        
        # Auto-refresh component
        auto_refresh_component = gr.Interval(5, interactive=False)
        
        # Refresh actions
        refresh_btn.click(get_current_status, outputs=[status_box, progress_bar, log_output])
        auto_refresh_component.change(
            fn=lambda: get_current_status() if auto_refresh.value else None,
            outputs=[status_box, progress_bar, log_output]
        )
        auto_refresh.change(lambda x: gr.update(interactive=x), inputs=auto_refresh, outputs=auto_refresh_component)
        
        # Initial status load
        demo.load(get_current_status, outputs=[status_box, progress_bar, log_output])

if __name__ == "__main__":
    demo.launch(share=True)