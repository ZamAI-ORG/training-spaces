# ZamAI Datasets

This directory contains sample datasets for training and fine-tuning ZamAI models:

1. **Translation Dataset** (`pashto_english_translation.json`)
   - Pashto to English translation pairs
   - Use with `zamai-translator-pashto-en` model

2. **Sentiment Analysis Dataset** (`pashto_sentiment.json`)
   - Pashto text with sentiment labels (positive/negative)
   - Use with `zamai-sentiment-pashto` model

3. **Question Answering Dataset** (`pashto_qa.json`)
   - Pashto questions with their answers
   - Use with `zamai-qa-pashto` model

4. **Chat Dataset** (`pashto_chat.json`)
   - Pashto conversation examples (user/assistant)
   - Use with `zamai-pashto-chat-8b` and other chat models

## How to Use

These datasets can be loaded in the respective model Spaces:

```python
import json

# Load a dataset
with open("path_to_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Use for training/fine-tuning
```
