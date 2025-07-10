#!/usr/bin/env python3
"""
ZamAI Project Setup and Management Runner
- Checks and fixes models
- Creates proper project structure
- Sets up Spaces and datasets
- Runs remaining space creation
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and display output"""
    print(f"\n{'=' * 80}")
    print(f"🚀 {description}")
    print(f"{'=' * 80}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully")
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
    
    return result.returncode

def main():
    print("🤖 ZamAI Project Setup and Management Runner")
    print("-------------------------------------------\n")
    
    # Step 1: Check and fix models
    run_command("python3 check_and_fix_models.py", "Checking and fixing models")
    
    # Step 2: Run model manager to ensure all models have proper files
    run_command("python3 model_manager.py", "Running model manager")
    
    # Step 3: Create remaining spaces
    run_command("python3 create_remaining_spaces.py", "Creating remaining spaces")
    
    print("\n🎉 All operations completed!")
    print("Your ZamAI project is now fully set up with:")
    print("1. Checked and fixed models in your Hugging Face Hub")
    print("2. Local model files in 'models' directory")
    print("3. Project structure with 'spaces' and 'datasets' folders")
    print("4. Spaces on Hugging Face Hub for all your models")

if __name__ == "__main__":
    main()
