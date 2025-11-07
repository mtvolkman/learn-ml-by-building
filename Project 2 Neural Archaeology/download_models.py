"""
Download models locally for Neural Archaeology project
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Configuration
MODELS_DIR = "./models"

# Get HuggingFace token from user
print("="*60)
print("HuggingFace Token Required")
print("="*60)
print("Get your token at: https://huggingface.co/settings/tokens")
print("You need 'Read' access for downloading models.")
print()
HF_TOKEN = input("Enter your HuggingFace token: ").strip()

if not HF_TOKEN:
    print("Error: No token provided. Exiting.")
    exit(1)

# Login to HuggingFace (needed for Gemma)
print("\nLogging into HuggingFace...")
try:
    login(token=HF_TOKEN)
    print("✓ Login successful")
except Exception as e:
    print(f"✗ Login failed: {e}")
    print("Please check your token and try again.")
    exit(1)

# Create models directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Model 1: SmolLM-1.7B-Instruct
print("\n" + "="*60)
print("Downloading SmolLM-1.7B-Instruct (~3.4 GB)")
print("="*60)
model_name1 = "HuggingFaceTB/SmolLM-1.7B-Instruct"
save_path1 = os.path.join(MODELS_DIR, "SmolLM-1.7B-Instruct")

try:
    print(f"Downloading tokenizer...")
    tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
    tokenizer1.save_pretrained(save_path1)
    print(f"✓ Tokenizer saved to {save_path1}")
    
    print(f"Downloading model (this may take a few minutes)...")
    model1 = AutoModelForCausalLM.from_pretrained(model_name1)
    model1.save_pretrained(save_path1)
    print(f"✓ Model saved to {save_path1}")
    
    # Clean up memory
    del model1, tokenizer1
    
except Exception as e:
    print(f"✗ Error downloading SmolLM: {e}")

# Model 2: Gemma-2-2b-it
print("\n" + "="*60)
print("Downloading Gemma-2-2b-it (~5 GB)")
print("="*60)
model_name2 = "google/gemma-2-2b-it"
save_path2 = os.path.join(MODELS_DIR, "gemma-2-2b-it")

try:
    print(f"Downloading tokenizer...")
    tokenizer2 = AutoTokenizer.from_pretrained(model_name2, token=HF_TOKEN)
    tokenizer2.save_pretrained(save_path2)
    print(f"✓ Tokenizer saved to {save_path2}")
    
    print(f"Downloading model (this may take a few minutes)...")
    model2 = AutoModelForCausalLM.from_pretrained(model_name2, token=HF_TOKEN)
    model2.save_pretrained(save_path2)
    print(f"✓ Model saved to {save_path2}")
    
    # Clean up memory
    del model2, tokenizer2
    
except Exception as e:
    print(f"✗ Error downloading Gemma: {e}")

print("\n" + "="*60)
print("Download Complete!")
print("="*60)
print(f"\nModels saved in: {os.path.abspath(MODELS_DIR)}")
print("\nTo use these models, update your load_model_and_tokenizer() calls:")
print(f"  - SmolLM: './models/SmolLM-1.7B-Instruct'")
print(f"  - Gemma: './models/gemma-2-2b-it'")
