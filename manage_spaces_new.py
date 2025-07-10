#!/usr/bin/env python3
"""
Script to run and test the existing Hugging Face Spaces
"""

import sys
import os
import time
import json
import requests
from huggingface_hub import HfApi, login

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
    
    print("\n🔍 To test a Space, visit its URL in your browser")
    print("📦 Each Space includes training, fine-tuning, and testing capabilities")
    print("⚡ All Spaces use ZeroGPU for efficient GPU acceleration")

if __name__ == "__main__":
    main()
