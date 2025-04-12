from datasets import load_dataset
import json
import os

# Load CodeSearchNet Python dataset (first 10K rows)
print("Loading dataset...")
dataset = load_dataset("code_search_net", "python", split="train[:10000]")

# Define chunking strategy
def chunk_code(code, chunk_size=512, overlap=256):
    if not code:
        return []
    return [code[i:i + chunk_size] for i in range(0, len(code), chunk_size - overlap)]

# Output file
output_file = "code_chunks.jsonl"
output_dir = os.path.dirname(output_file)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Process and write chunks
print("Processing and chunking...")
with open(output_file, "w") as f:
    chunk_count = 0
    for entry in dataset:
        code = entry.get("func_code_string", "")
        docstring = entry.get("docstring", "")
        func_name = entry.get("func_name", "")
        identifier = entry.get("identifier", "")
        sha = entry.get("sha", "")
        
        chunks = chunk_code(code)
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                chunk_data = {
                    "chunk_id": f"{identifier}_{i}",
                    "code_chunk": chunk,
                    "docstring": docstring,
                    "func_name": func_name,
                    "sha": sha
                }
                json.dump(chunk_data, f)
                f.write("\n")
                chunk_count += 1

print(f"✅ Saved {chunk_count} code chunks to {output_file}.")
