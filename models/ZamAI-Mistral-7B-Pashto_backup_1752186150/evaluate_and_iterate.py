#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iterative evaluation script for fine-tuned models.
This script helps evaluate model performance on Pashto text
and provides guidance for iterative improvements.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and iteratively improve a fine-tuned model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name or local path",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="zamai_pashto_dataset.json",
        help="JSON file with the dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_evaluation_results.json",
        help="File to save evaluation results",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights with the base model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()

def categorize_samples(data):
    """Categorize dataset samples by type and length for better analysis"""
    categories = defaultdict(list)
    length_bins = {
        "short": (0, 50),
        "medium": (50, 200),
        "long": (200, float('inf'))
    }
    
    for idx, item in enumerate(data):
        # Skip samples without input or output
        if not item.get('input') or not item.get('output'):
            continue
        
        # Determine length category
        input_len = len(item['input'])
        for length_cat, (min_len, max_len) in length_bins.items():
            if min_len <= input_len < max_len:
                categories[length_cat].append(idx)
                break
                
    return categories

def evaluate_model(model, tokenizer, data, sample_indices, max_new_tokens, temperature):
    """Evaluate model on selected samples"""
    results = []
    
    for idx in tqdm(sample_indices, desc="Evaluating samples"):
        try:
            item = data[idx]
            input_text = item['input']
            reference_output = item['output']
            
            # Record start time
            start_time = time.time()
            
            # Generate text
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output if it's there
            if model_output.startswith(input_text):
                model_output = model_output[len(input_text):].strip()
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            results.append({
                "index": idx,
                "input": input_text,
                "reference": reference_output,
                "model_output": model_output,
                "time_taken": time_taken,
                "output_length": len(model_output)
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            
    return results

def analyze_results(results):
    """Analyze evaluation results and provide recommendations"""
    if not results:
        return "No results to analyze"
    
    analysis = {}
    
    # Calculate statistics
    output_lengths = [r["output_length"] for r in results]
    time_taken = [r["time_taken"] for r in results]
    
    analysis["statistics"] = {
        "avg_output_length": sum(output_lengths) / len(output_lengths),
        "min_output_length": min(output_lengths),
        "max_output_length": max(output_lengths),
        "avg_time_per_token": sum(time_taken) / sum(output_lengths) if sum(output_lengths) > 0 else 0,
        "total_samples": len(results)
    }
    
    # Generate recommendations
    recommendations = []
    
    # Check if outputs are too short
    if analysis["statistics"]["avg_output_length"] < 30:
        recommendations.append(
            "The model's outputs are very short. Try fine-tuning with more diverse examples "
            "or adjusting the temperature parameter for generation."
        )
    
    # Check completion rate (how many reached max tokens)
    max_token_samples = sum(1 for r in results if r["output_length"] >= 0.9 * args.max_new_tokens)
    if max_token_samples > 0.5 * len(results):
        recommendations.append(
            f"{max_token_samples} samples reached near max token length, which might indicate truncated outputs. "
            f"Consider increasing max_new_tokens for evaluation."
        )
    
    # Analyze language balance in outputs
    pashto_chars = sum(1 for r in results for c in r["model_output"] if '\u0600' <= c <= '\u06FF')
    latin_chars = sum(1 for r in results for c in r["model_output"] if 'a' <= c.lower() <= 'z')
    
    if latin_chars > pashto_chars * 0.5:
        recommendations.append(
            "Model outputs contain substantial Latin script. For a Pashto model, consider "
            "more Pashto examples in your training data or longer fine-tuning."
        )
    
    # Add recommendations to analysis
    analysis["recommendations"] = recommendations
    
    # Identify areas for adding more examples
    failing_areas = []
    # This is a simplified analysis - in a real scenario, you might use embeddings or clustering
    if any(len(r["model_output"]) < 20 for r in results):
        failing_areas.append("short responses")
    
    analysis["areas_for_data_augmentation"] = failing_areas
    
    return analysis

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_file}")
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Dataset loaded with {len(data)} examples")
    
    # Categorize samples
    print("Categorizing samples...")
    categories = categorize_samples(data)
    for category, indices in categories.items():
        print(f"  {category}: {len(indices)} examples")
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.merge_lora:
        print("Loading and merging LoRA model...")
        base_model_name = None  # You'd need to specify the base model or extract it from config
        if not base_model_name:
            print("For merging LoRA models, please specify the base model name in the script.")
            return
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.model_name)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    # Select samples from each category
    all_sample_indices = []
    samples_per_category = max(1, args.num_samples // len(categories))
    
    for category, indices in categories.items():
        if indices:
            # Take at least one sample from each category, up to samples_per_category
            category_samples = random.sample(indices, min(samples_per_category, len(indices)))
            all_sample_indices.extend(category_samples)
    
    # If we need more samples to reach args.num_samples, take random ones
    if len(all_sample_indices) < args.num_samples:
        remaining_indices = list(set(range(len(data))) - set(all_sample_indices))
        if remaining_indices:
            additional_samples = random.sample(
                remaining_indices, 
                min(args.num_samples - len(all_sample_indices), len(remaining_indices))
            )
            all_sample_indices.extend(additional_samples)
    
    print(f"Selected {len(all_sample_indices)} samples for evaluation")
    
    # Evaluate model
    results = evaluate_model(
        model, tokenizer, data, all_sample_indices, 
        args.max_new_tokens, args.temperature
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_results(results)
    
    # Save results
    output = {
        "model_name": args.model_name,
        "evaluation_samples": len(results),
        "results": results,
        "analysis": analysis
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output_file}")
    
    # Display summary
    print("\n===== Evaluation Summary =====")
    print(f"Model: {args.model_name}")
    print(f"Samples evaluated: {len(results)}")
    print("\nStatistics:")
    for key, value in analysis["statistics"].items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    if analysis["recommendations"]:
        print("\nRecommendations for improvement:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"{i}. {rec}")
    
    print("\nIterative Process Guide:")
    print("1. Review the generated outputs in the results file")
    print("2. Based on recommendations, consider:")
    print("   - Adjusting training parameters (learning rate, epochs)")
    print("   - Adding more examples to underrepresented categories")
    print("   - Cleaning existing training data")
    print("3. Update your dataset and start a new training run")
    print("4. Evaluate again and compare results")
    
if __name__ == "__main__":
    main()
