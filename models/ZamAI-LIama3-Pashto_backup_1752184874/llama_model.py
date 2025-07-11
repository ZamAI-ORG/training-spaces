from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from .model_config import MODEL_CONFIG

class ZamAIPashtoModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_base_model(self):
        """Load the base Llama 3.1 model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["base_model"])
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG["base_model"],
                device_map="auto",
                load_in_8bit=True
            )
            return True
        except Exception as e:
            print(f"Error loading base model: {str(e)}")
            return False

    def load_pashto_model(self):
        """Load your fine-tuned Pashto model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["fine_tuned_model"])
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_CONFIG["fine_tuned_model"],
                device_map="auto",
                load_in_8bit=True
            )
            return True
        except Exception as e:
            print(f"Error loading Pashto model: {str(e)}")
            return False

    def setup_pipeline(self):
        """Set up the generation pipeline"""
        if not (self.model and self.tokenizer):
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=100,
            temperature=0.7,
            device=self.device
        )

    def generate_text(self, prompt, max_length=100, temperature=0.7):
        if not self.generator:
            raise RuntimeError("Pipeline not initialized. Call setup_pipeline() first.")
        
        try:
            response = self.generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return response[0]['generated_text']
        except Exception as e:
            print(f"Error generating text: {str(e)}")
            return None

    def cleanup(self):
        if self.model:
            del self.model
            del self.generator
            torch.cuda.empty_cache()