import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceModel, SentenceTransformer

# 1. Sample Corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence and machine learning transform search engines.",
    "BM25 is a strong lexical ranking function for keywords.",
    "Vector embeddings capture deep semantic meaning in text.",
]

# Tokenize corpus for BM25
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# 2. Setup Vector Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus)


def vector_search(query, top_k=3):
  query_vector = model.encode(query)
  # Compute cosine similarity
  scores = np.dot(corpus_embeddings, query_vector) / (
      np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_vector)
  )
  top_indices = np.argsort(scores)[::-1][:top_k]
  return [(int(idx), float(scores[idx])) for idx in top_indices]


def bm25_search(query, top_k=3):
  tokenized_query = query.lower().split()
  scores = bm25.get_scores(tokenized_query)
  top_indices = np.argsort(scores)[::-1][:top_k]
  return [(int(idx), float(scores[idx])) for idx in top_indices]


# 3. Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(bm25_res, vector_res, k=60):
  rrf_scores = {}

  for rank, (doc_idx, _) in enumerate(bm25_res):
    rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

  for rank, (doc_idx, _) in enumerate(vector_res):
    rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)

  sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
  return sorted_docs


# 4. Execute Hybrid Query
query = "How do vector embeddings work?"
bm25_results = bm25_search(query, top_k=3)
vector_results = vector_search(query, top_k=3)
hybrid_results = reciprocal_rank_fusion(bm25_results, vector_results)

print("Final RRF Hybrid Rankings (Doc Index, Score):")
for doc_idx, score in hybrid_results:
  print(f"Doc {doc_idx}: {corpus[doc_idx]} (Score: {score:.4f})")
