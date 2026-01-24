
import os
import sys
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset

def push_dataset(file_path, repo_id, token=None):
    """
    Pushes a JSONL dataset to Hugging Face Hub
    """
    print(f"🚀 Loading dataset from {file_path}...")
    try:
        dataset = load_dataset('json', data_files=file_path)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print(f"📦 Pushing to {repo_id}...")
    try:
        dataset.push_to_hub(repo_id, token=token)
        print(f"✅ Successfully pushed to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Error pushing to Hub: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python push_dataset.py <file_path> <repo_id>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    repo_id = sys.argv[2]
    
    push_dataset(file_path, repo_id)
