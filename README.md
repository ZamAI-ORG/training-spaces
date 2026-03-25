# Training Spaces

Reusable training and experiment spaces for ZamAI Labs (templates, scripts, and reproducible runs).

- Website: https://zamai.dev
- Labs: https://github.com/ZamAI-ORG


# ZamAI Models and Training Spaces

This repository contains scripts to manage Hugging Face models and spaces for the ZamAI project.

## Features

- 📤 Upload models to Hugging Face Hub
- 🗑️ Delete existing Hugging Face spaces
- 🏗️ Create new spaces with train/finetune/test functionality
- ⚡ All spaces use ZeroGPU for efficient computation

## Models Included

Your models directory contains the following models:
- Multilingual-ZamAI-Embeddings
- ZamAI-LIama3-Pashto
- ZamAI-Mistral-7B-Pashto
- ZamAI-Phi-3-Mini-Pashto
- ZamAI-Whisper-v3-Pashto
- pashto-base-bloom
- zamai-dialogpt-pashto-v3
- zamai-pashto-chat-8b
- zamai-qa-pashto
- zamai-sentiment-pashto
- zamai-translator-pashto-en

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Get your Hugging Face token from: https://huggingface.co/settings/tokens

### Usage

#### Option 1: Interactive Management (Recommended)
```bash
python manage_all.py
```

This interactive script will guide you through:
1. Uploading models to Hugging Face Hub
2. Deleting existing spaces
3. Creating new spaces with train/finetune/test functionality

#### Option 2: Individual Scripts

##### Upload Models to Hub
```bash
python upload_models_to_hub.py --models-dir ./models --username YOUR_USERNAME
```

##### Manage Spaces
```bash
# Delete all existing spaces
python manage_spaces.py --delete-all --username YOUR_USERNAME

# Create new spaces for all models
python manage_spaces.py --create-spaces --username YOUR_USERNAME

# Create spaces for specific models
python manage_spaces.py --create-spaces --models model1 model2 --username YOUR_USERNAME
```

## Space Features

Each created space will have:

### 🧪 Test Tab
- Input text field for testing the model
- Adjustable max length and temperature parameters
- Real-time generation with the model

### 🏋️ Train Tab
- Training dataset input
- Configurable epochs and learning rate
- Training from scratch functionality

### 🔧 Fine-tune Tab
- Fine-tuning dataset input
- Configurable parameters for fine-tuning
- Fine-tune existing models

## ZeroGPU Integration

All spaces are configured to use ZeroGPU (`hardware: zero-a10g`) for efficient GPU computation without additional costs.

## File Structure

```
.
├── manage_all.py           # Master script for interactive management
├── upload_models_to_hub.py # Upload models to Hugging Face Hub
├── manage_spaces.py        # Manage Hugging Face Spaces
├── requirements.txt        # Python dependencies
├── models/                 # Your model directories
└── spaces/                # Generated space files (created during execution)
```

## Authentication

You'll need to provide:
1. **Hugging Face Token**: Get from https://huggingface.co/settings/tokens
2. **Username**: Your Hugging Face username

## Troubleshooting

### Common Issues

1. **Empty model directories**: Ensure your model files are present in the respective directories
2. **Authentication errors**: Verify your Hugging Face token has write permissions
3. **Rate limiting**: The scripts include delays to respect API rate limits

### Support

For issues specific to ZamAI models, please check the individual model documentation on Hugging Face Hub.

## License

This project is licensed under the Apache License 2.0.
