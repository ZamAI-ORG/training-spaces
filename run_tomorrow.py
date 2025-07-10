#!/usr/bin/env python3
"""
Simple script to run tomorrow - creates remaining spaces with proper ZeroGPU
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 Creating remaining Hugging Face Spaces...")
    print("This script will create the 7 remaining spaces with:")
    print("  - ✅ Proper ZeroGPU configuration")
    print("  - 🧪 Full testing capabilities") 
    print("  - 🏋️ Real training implementation")
    print("  - 🔧 Actual fine-tuning features")
    print("  - 📤 Model upload functionality")
    print()
    
    # Import and run the remaining spaces script
    try:
        exec(open('create_remaining_spaces.py').read())
    except FileNotFoundError:
        print("❌ create_remaining_spaces.py not found in current directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
