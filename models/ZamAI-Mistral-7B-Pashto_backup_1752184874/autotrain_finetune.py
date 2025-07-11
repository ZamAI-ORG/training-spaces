#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to start a Mistral-7B fine-tuning job on Hugging Face AutoTrain
This uses the AutoTrain API to launch a fine-tuning job in the cloud
"""

import argparse
import os
from autotrain.trainers.clm.__main__ import train

def parse_args():
    parser = argparse.ArgumentParser(description="Start a fine-tuning job on Hugging Face AutoTrain")
    parser.add_argument(
        "--project_name",
        type=str,
        default="ZamAI-Mistral-7B-Pashto",
        help="Name for your AutoTrain project"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tasal9/ZamAI_Pashto_Training",
        help="Dataset on HF Hub to use for training"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name containing the formatted text"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA attention dropout"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face API token (required)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.hf_token:
        print("Error: Hugging Face API token is required")
        print("Get your token from https://huggingface.co/settings/tokens")
        return
    
    print(f"Starting fine-tuning job for {args.model} on dataset {args.dataset}")
    
    # Define the training configuration
    config = {
        "model": args.model,
        "data_path": args.dataset,
        "text_column": args.text_column,
        "project_name": args.project_name,
        "token": args.hf_token,
        "lr": args.lr,
        "epochs": args.epochs,
        "push_to_hub": True,
        "repo_id": f"tasal9/{args.project_name}",
        "trainer": "sft",
        "peft": True,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "batch_size": 4,
        "block_size": 1024,
        "logging_steps": 10,
        "target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "log": "wandb"  # You can change to "tensorboard" or "none"
    }
    
    # Run the training job
    train(config)
    
    print("\n==== Training Job Submitted ====")
    print(f"Project: {args.project_name}")
    print(f"You can monitor your training job at: https://ui.autotrain.huggingface.co/projects")
    print(f"Your fine-tuned model will be pushed to: https://huggingface.co/tasal9/{args.project_name}")

if __name__ == "__main__":
    main()
