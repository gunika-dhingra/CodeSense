from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Parameters
chunk_file = "code_chunks.jsonl"
embedding_model = "all-MiniLM-L6-v2"
output_index_file = "code_faiss_index.bin"
metadata_file = "metadata.json"

# Load chunks
print("Loading code chunks...")
texts = []
metadata = []

with open(chunk_file, "r") as f:
    for line in f:
        obj = json.loads(line)
        combined_text = f"{obj.get('docstring', '')}\n{obj.get('code_chunk', '')}".strip()
        if combined_text:
            texts.append(combined_text)
            metadata.append({
                "chunk_id": obj.get("chunk_id"),
                "func_name": obj.get("func_name"),
                "sha": obj.get("sha")
            })

print(f"Loaded {len(texts)} chunks for embedding.")

# Load model and generate embeddings
print("Generating embeddings...")
model = SentenceTransformer(embedding_model)
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)  # Use normalize for cosine

# Create FAISS index (cosine similarity = dot product of normalized vectors)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
index.add(embeddings)
faiss.write_index(index, output_index_file)

# Save metadata for lookup later
with open(metadata_file, "w") as f:
    json.dump(metadata, f)

print(f"✅ FAISS index saved to '{output_index_file}' with {index.ntotal} vectors.")
print(f"✅ Metadata saved to '{metadata_file}'.")
