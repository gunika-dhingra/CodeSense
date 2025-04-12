from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import json
import re
from collections import defaultdict

# Load FAISS index
index = faiss.read_index("code_faiss_index.bin")

# Load metadata
with open("metadata.json", "r") as f:
    metadata = json.load(f)

# Load code chunks
code_chunks = []
chunk_lookup = {}
for line in open("code_chunks.jsonl", "r"):
    obj = json.loads(line)
    chunk_lookup[obj["chunk_id"]] = obj
    code_chunks.append(obj)

# Build BM25 corpus
bm25_corpus = []
chunk_id_list = []
for obj in code_chunks:
    text = f"{obj.get('docstring', '')} {obj.get('code_chunk', '')}"
    tokens = re.findall(r'\w+', text.lower())
    bm25_corpus.append(tokens)
    chunk_id_list.append(obj["chunk_id"])

bm25 = BM25Okapi(bm25_corpus)
dense_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reciprocal Rank Fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    fusion_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            fusion_scores[doc_id] += 1 / (k + rank)
    sorted_docs = sorted(fusion_scores.items(), key=lambda x: -x[1])
    return [doc_id for doc_id, _ in sorted_docs]

# Query Function
def hybrid_query(query, top_k=5, alpha=0.5, use_rrf=True):
    query_embedding = dense_model.encode(query, normalize_embeddings=True).reshape(1, -1)
    dense_scores, dense_indices = index.search(query_embedding, top_k * 2)
    dense_results = [metadata[i]["chunk_id"] for i in dense_indices[0] if i < len(metadata)]

    bm25_tokens = re.findall(r'\w+', query.lower())
    sparse_scores = bm25.get_scores(bm25_tokens)
    sparse_ranked = np.argsort(sparse_scores)[::-1][:top_k * 2]
    sparse_results = [chunk_id_list[i] for i in sparse_ranked]

    if use_rrf:
        ranked_lists = [sparse_results, dense_results]
        fused_ids = reciprocal_rank_fusion(ranked_lists)[:top_k]
    else:
        fused_ids = dense_results[:top_k]

    results = []
    for cid in fused_ids:
        chunk = chunk_lookup[cid]
        results.append({
            "chunk_id": cid,
            "func_name": chunk.get("func_name"),
            "docstring": chunk.get("docstring", "")[:100],
            "code_chunk": chunk.get("code_chunk", "")[:300]
        })
    return results

# Run a test query
def save_results_to_txt(query, results, file_path="query_responses_top5.txt"):
    try:
        with open(file_path, "w") as f:
            f.write(f"Query: {query}\n\n")
            for i, res in enumerate(results, 1):
                f.write(f"Function {i}:\n")
                f.write(f"{res['code_chunk']}\n\n")
        print(f"✅ Results saved to {file_path}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
query = "reverse a linked list"
top_results = hybrid_query(query, top_k=5)
save_results_to_txt(query,top_results)
for i, res in enumerate(top_results, 1):
    print(f"\n🔹 Result #{i}")
    print(f"Chunk ID   : {res['chunk_id']}")
    print(f"Function   : {res['func_name']}")
    print(f"Docstring  : {res['docstring']}")
    print(f"Code Chunk :\n{res['code_chunk']}")

