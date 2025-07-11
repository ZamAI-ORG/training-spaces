#!/usr/bin/env python3
"""
Advanced Space Manager - Run, Test, and Upgrade existing Hugging Face Spaces
Adds "Load Model" button requirement and enhanced features
"""

import sys
import os
import time
import json
import requests
import shutil
from huggingface_hub import HfApi, login, Repository

def load_credentials():
    """Load credentials from HF-Credentials.txt"""
    try:
        with open("HF-Credentials.txt", "r") as f:
            lines = f.readlines()
        username = lines[0].split(":")[1].strip().strip("<>")
        token = lines[1].split(":")[1].strip().strip("<>")
        return username, token
    except Exception as e:
        print(f"❌ Failed to load credentials: {e}")
        sys.exit(1)

def list_user_spaces(username, token):
    """List all spaces for a user"""
    api = HfApi()
    try:
        spaces = list(api.list_spaces(author=username, token=token))
        return spaces
    except Exception as e:
        print(f"❌ Failed to list spaces: {e}")
        return []

def check_space_status(space_id, token):
    """Check if a space is running"""
    url = f"https://huggingface.co/api/spaces/{space_id}/runtime"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("stage", "UNKNOWN")
        return "ERROR"
    except Exception as e:
        print(f"❌ Failed to check space status: {e}")
        return "ERROR"

def restart_space(space_id, token):
    """Restart a Hugging Face Space"""
    url = f"https://huggingface.co/api/spaces/{space_id}/restart"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(url, headers=headers)
        if response.status_code == 200:
            print(f"✅ Successfully restarted {space_id}")
            return True
        else:
            print(f"❌ Failed to restart {space_id}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Failed to restart {space_id}: {e}")
        return False

def wait_for_space_running(space_id, token, max_wait=300):
    """Wait for a space to be in the RUNNING stage"""
    start_time = time.time()
    print(f"⏳ Waiting for {space_id} to be ready...")
    
    while time.time() - start_time < max_wait:
        stage = check_space_status(space_id, token)
        
        if stage == "RUNNING":
            print(f"✅ {space_id} is now running!")
            return True
        elif stage in ["BUILDING", "STARTING"]:
            print(f"⏳ {space_id} is {stage.lower()}... ({int(time.time() - start_time)}s)")
            time.sleep(10)
        else:
            print(f"❌ {space_id} is in unexpected state: {stage}")
            return False
    
    print(f"❌ Timeout waiting for {space_id} to be ready")
    return False

def test_space_endpoint(space_id):
    """Test a Space endpoint with a simple request"""
    # Simple test just checking if the space is accessible
    url = f"https://{space_id.replace('/', '-')}.hf.space/"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ {space_id} is accessible")
            return True
        else:
            print(f"❌ {space_id} returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to access {space_id}: {e}")
        return False

def update_space_app(space_id, model_id, token):
    """Update a Space with the advanced template"""
    space_name = space_id
    model_name = model_id.split('/')[-1]
    
    # Get model type
    model_type = "causal_lm"  # Default
    if "sentiment" in model_name.lower():
        model_type = "text_classification"
    elif "translator" in model_name.lower() or "seq2seq" in model_name.lower():
        model_type = "seq2seq"
    
    # Create a temporary directory
    temp_dir = f"temp_{model_name}_space_update"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    try:
        # Clone the space repo
        repo = Repository(local_dir=temp_dir, clone_from=f"https://huggingface.co/spaces/{space_name}")
        
        # Read the template
        with open("advanced_space_template.py", "r") as f:
            template_content = f.read()
        
        # Replace placeholders
        app_content = template_content.replace("MODEL_NAME_PLACEHOLDER", model_id)
        app_content = app_content.replace("MODEL_TYPE_PLACEHOLDER", model_type)
        
        # Create README.md
        readme_content = f"""---
title: {model_name} Advanced Training Space
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-gpu-a10g
---

# {model_name} Advanced Training Space

This space provides enhanced functionality for working with the {model_name} model:

## ✨ New Features

1. **Load Model Button**: Explicitly load the model before using other features
2. **Advanced Generation Settings**: Control temperature, top-p, and repetition penalty
3. **Model Evaluation**: Measure model performance on test data
4. **Enhanced Training**: Better progress tracking and parameter tuning
5. **Model Information**: View details about the model architecture and parameters
6. **Recommendations**: Get suggestions for next steps after each operation

## 🔧 Capabilities

- **Test**: Generate text with customizable parameters
- **Train**: Train or fine-tune the model with your data
- **Evaluate**: Measure model performance quantitatively
- **Upload**: Save your trained models to Hugging Face Hub

Powered by ZeroGPU for efficient GPU acceleration.
"""
        
        # Create requirements.txt
        requirements_content = """gradio>=4.36.1
spaces
torch>=2.0.0
transformers>=4.30.0
datasets>=2.13.0
huggingface_hub>=0.16.0
numpy>=1.24.0
accelerate>=0.21.0
scikit-learn>=1.2.2
"""
        
        # Write the files
        with open(os.path.join(temp_dir, "app.py"), "w") as f:
            f.write(app_content)
        
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        with open(os.path.join(temp_dir, "requirements.txt"), "w") as f:
            f.write(requirements_content)
        
        # Push to the space
        repo.git_add(auto_lfs_track=True)
        repo.git_commit("Update Space with advanced template including Load Model button and enhanced features")
        repo.git_push()
        
        print(f"✅ Successfully updated {space_name} with advanced template")
        shutil.rmtree(temp_dir)  # Clean up
        return True
        
    except Exception as e:
        print(f"❌ Failed to update {space_name}: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Clean up on error
        return False

def run_spaces(spaces, token, restart=True):
    """Run and test all spaces"""
    results = {}
    
    for i, space in enumerate(spaces, 1):
        space_id = space.id
        print(f"\n[{i}/{len(spaces)}] Processing {space_id}...")
        
        # Check current status
        status = check_space_status(space_id, token)
        print(f"📊 Current status: {status}")
        
        # Restart if requested and not already running
        if restart and status != "RUNNING":
            restart_space(space_id, token)
            running = wait_for_space_running(space_id, token)
        elif status == "RUNNING":
            running = True
            print(f"✅ {space_id} is already running")
        else:
            running = False
        
        # Test the space if it's running
        if running:
            accessible = test_space_endpoint(space_id)
        else:
            accessible = False
        
        # Store results
        results[space_id] = {
            "status": check_space_status(space_id, token),
            "running": running,
            "accessible": accessible
        }
        
        print(f"📋 Summary for {space_id}:")
        print(f"  Status: {results[space_id]['status']}")
        print(f"  Running: {'✅' if results[space_id]['running'] else '❌'}")
        print(f"  Accessible: {'✅' if results[space_id]['accessible'] else '❌'}")
        
        # URL to access the space
        if results[space_id]['accessible']:
            print(f"  🌐 URL: https://{space_id.replace('/', '-')}.hf.space/")
    
    return results

def update_all_spaces(spaces, username, token):
    """Update all spaces with advanced template"""
    success_count = 0
    failed_count = 0
    
    for i, space in enumerate(spaces, 1):
        space_id = space.id
        print(f"\n[{i}/{len(spaces)}] Updating {space_id} with advanced template...")
        
        # Extract model ID from space name
        model_name = space_id.replace(f"{username}/", "").replace("-space", "")
        model_id = f"{username}/{model_name}"
        
        # Update the space
        success = update_space_app(space_id, model_id, token)
        
        if success:
            success_count += 1
            print(f"✅ Successfully updated {space_id}")
            
            # Restart the space to apply changes
            print(f"🔄 Restarting {space_id} to apply changes...")
            restart_space(space_id, token)
            wait_for_space_running(space_id, token)
        else:
            failed_count += 1
            print(f"❌ Failed to update {space_id}")
        
        # Add delay to avoid rate limits
        if i < len(spaces):
            print("⏳ Waiting 10 seconds before next update...")
            time.sleep(10)
    
    print(f"\n📊 Space Update Summary:")
    print(f"  ✅ Successfully updated: {success_count}/{len(spaces)}")
    print(f"  ❌ Failed to update: {failed_count}/{len(spaces)}")
    
    return success_count, failed_count

def main():
    # Load credentials
    username, token = load_credentials()
    
    try:
        login(token=token)
        print(f"✅ Successfully authenticated as {username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    # List spaces
    print("\n🔍 Fetching your Spaces from Hugging Face Hub...")
    spaces = list_user_spaces(username, token)
    
    if not spaces:
        print("❌ No Spaces found on your Hugging Face account")
        sys.exit(1)
    
    print(f"📦 Found {len(spaces)} Spaces:")
    for i, space in enumerate(spaces, 1):
        print(f"  {i}. {space.id}")
    
    # Menu
    print("\n🔧 Space Management Options:")
    print("1. Run and test existing Spaces")
    print("2. Update Spaces with advanced template (adds Load Model button)")
    print("3. Both: Update and then run Spaces")
    print("4. Exit")
    
    choice = input("\nSelect an option (1-4): ").strip()
    
    if choice == "1" or choice == "3":
        # Run all spaces
        print("\n🚀 Running and testing all Spaces...")
        results = run_spaces(spaces, token)
        
        # Save results
        with open("spaces_status.json", "w") as f:
            json.dump({k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
        
        print(f"\n📝 Results saved to spaces_status.json")
        
        # Count successes
        running_count = sum(1 for space in results.values() if space["running"])
        accessible_count = sum(1 for space in results.values() if space["accessible"])
        
        print(f"\n📊 Final Results:")
        print(f"✅ Running: {running_count}/{len(spaces)} Spaces")
        print(f"✅ Accessible: {accessible_count}/{len(spaces)} Spaces")
        
        # Show URLs for all accessible spaces
        if accessible_count > 0:
            print("\n🌐 Space URLs:")
            for space_id, result in results.items():
                if result["accessible"]:
                    print(f"  - {space_id}: https://{space_id.replace('/', '-')}.hf.space/")
    
    if choice == "2" or choice == "3":
        # Check if advanced_space_template.py exists
        if not os.path.exists("advanced_space_template.py"):
            print("❌ advanced_space_template.py not found! Please make sure this file exists.")
            sys.exit(1)
            
        # Update all spaces
        print("\n🔄 Updating all Spaces with advanced template...")
        print("This will add a Load Model button and enhanced features to all Spaces")
        confirm = input("Continue? (y/n): ").strip().lower()
        
        if confirm == "y":
            success_count, failed_count = update_all_spaces(spaces, username, token)
            
            if success_count > 0:
                print("\n✅ Successfully updated Spaces!")
                print("New features added:")
                print("  - Load Model button (must be clicked before using other features)")
                print("  - Advanced generation parameters (temperature, top-p, repetition penalty)")
                print("  - Model evaluation functionality")
                print("  - Enhanced training with progress tracking")
                print("  - Model information display")
                print("  - Recommendations for next steps")
        else:
            print("\n❌ Update cancelled")
    
    if choice == "4":
        print("\n👋 Exiting...")
        sys.exit(0)
    
    if choice not in ["1", "2", "3", "4"]:
        print("\n❌ Invalid choice")
        sys.exit(1)
        
    print("\n🔍 To test a Space, visit its URL in your browser")
    print("📦 Each Space now includes Load Model button and advanced features")
    print("⚡ All Spaces use ZeroGPU for efficient GPU acceleration")
    print("\n💡 RECOMMENDATIONS:")
    print("1. First click 'Load Model' button when using any Space")
    print("2. After loading, explore the model information to understand its capabilities")
    print("3. Test with simple prompts before trying more complex tasks")
    print("4. Use the evaluation tab to quantitatively measure model performance")

if __name__ == "__main__":
    main()
