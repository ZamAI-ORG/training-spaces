#!/bin/bash

echo "🚀 Starting ZamAI Mistral model deployment..."

# Change to the correct directory
cd /workspaces/Huggingface/ZamAI-Mistral-7B-Pashto

# Check if we have the HF token
if [ ! -f "../Multilingual-ZamAI-Embeddings/HF-Token.txt" ]; then
    echo "❌ HF Token file not found"
    exit 1
fi

# Read the token
HF_TOKEN=$(cat ../Multilingual-ZamAI-Embeddings/HF-Token.txt)

echo "✅ Token loaded successfully"

# Login to HF CLI
echo $HF_TOKEN | huggingface-cli login --token $HF_TOKEN

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Set up HF remote
echo "🔗 Setting up Hugging Face remote..."
git remote remove origin 2>/dev/null || true
git remote add origin https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto

# Add all files
echo "📤 Adding files..."
git add .

# Commit
echo "💾 Committing files..."
git commit -m "Upload ZamAI-Mistral-7B-Pashto model repository" || echo "Nothing to commit"

# Set main branch
git branch -M main

# Push to HF
echo "🚀 Pushing to Hugging Face Hub..."
git push origin main --force

echo "🎉 Successfully deployed to https://huggingface.co/tasal9/ZamAI-Mistral-7B-Pashto"
echo "Your Mistral model is now live on Hugging Face Hub!"
