from model.llama_model import ZamAIPashtoModel
from utils.auth import setup_huggingface_auth

def main():
    # Set up authentication
    HUGGINGFACE_TOKEN = "your_token_here"  # Replace with your actual token
    if not setup_huggingface_auth(HUGGINGFACE_TOKEN):
        print("Authentication failed")
        return

    # Initialize model
    model = ZamAIPashtoModel()

    try:
        # Load base Llama model first
        print("Loading base Llama 3.1 model...")
        if not model.load_base_model():
            print("Failed to load base model")
            return

        # Load your Pashto fine-tuned model
        print("Loading Pashto fine-tuned model...")
        if not model.load_pashto_model():
            print("Failed to load Pashto model")
            return

        # Set up the pipeline
        model.setup_pipeline()

        # Test generation
        prompt = "Write in Pashto: Hello, how are you?"
        response = model.generate_text(prompt)
        print(f"Generated text:\n{response}")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
    finally:
        model.cleanup()

if __name__ == "__main__":
    main()