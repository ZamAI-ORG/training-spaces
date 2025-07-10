#!/usr/bin/env python3
"""
Master script to handle all operations: upload models, delete spaces, create new spaces
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, args=None):
    """Run a Python script with optional arguments"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    print("🚀 ZamAI Models and Spaces Management Tool")
    print("=" * 50)
    
    # Get models from directory
    models_dir = Path("./models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        sys.exit(1)
    
    model_names = [d.name for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_names:
        print("❌ No model directories found")
        sys.exit(1)
    
    print(f"📦 Found {len(model_names)} models:")
    for i, model in enumerate(model_names, 1):
        print(f"  {i}. {model}")
    
    print("\\n🔧 What would you like to do?")
    print("1. Upload models to Hugging Face Hub")
    print("2. Delete all existing spaces")
    print("3. Create new spaces for models")
    print("4. Complete workflow (upload models + delete spaces + create new spaces)")
    print("5. Exit")
    
    choice = input("\\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        print("\\n📤 Uploading models to Hugging Face Hub...")
        success = run_script("upload_models_to_hub.py", ["--models-dir", "./models"])
        if success:
            print("✅ Models upload completed")
        else:
            print("❌ Models upload failed")
    
    elif choice == "2":
        print("\\n🗑️  Deleting all existing spaces...")
        success = run_script("manage_spaces.py", ["--delete-all"])
        if success:
            print("✅ Spaces deletion completed")
        else:
            print("❌ Spaces deletion failed")
    
    elif choice == "3":
        print("\\n🏗️  Creating new spaces...")
        success = run_script("manage_spaces.py", ["--create-spaces", "--models"] + model_names)
        if success:
            print("✅ Spaces creation completed")
        else:
            print("❌ Spaces creation failed")
    
    elif choice == "4":
        print("\\n🔄 Running complete workflow...")
        
        # Step 1: Upload models
        print("\\n📤 Step 1: Uploading models to Hugging Face Hub...")
        success1 = run_script("upload_models_to_hub.py", ["--models-dir", "./models"])
        
        if not success1:
            print("❌ Failed at step 1. Stopping workflow.")
            return
        
        # Step 2: Delete existing spaces
        print("\\n🗑️  Step 2: Deleting all existing spaces...")
        success2 = run_script("manage_spaces.py", ["--delete-all"])
        
        if not success2:
            print("❌ Failed at step 2. Continuing with step 3...")
        
        # Step 3: Create new spaces
        print("\\n🏗️  Step 3: Creating new spaces...")
        success3 = run_script("manage_spaces.py", ["--create-spaces", "--models"] + model_names)
        
        if success1 and success3:
            print("\\n✅ Complete workflow finished successfully!")
            print("\\n📋 Next steps:")
            print("1. Check the ./spaces/ directory for generated space files")
            print("2. Upload the space files to their respective Hugging Face spaces")
            print("3. Your models should now be available on Hugging Face Hub")
        else:
            print("\\n⚠️  Workflow completed with some issues. Check the logs above.")
    
    elif choice == "5":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
