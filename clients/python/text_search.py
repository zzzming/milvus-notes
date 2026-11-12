"""
Text semantic search demo: embed documents with sentence-transformers,
ingest into Milvus, then search by a natural-language query.

Closest to a real RAG ingestion + retrieval flow.

Usage: python text_search.py
First run downloads the embedding model (~90 MB).
"""

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

MILVUS_URI  = "http://localhost:19530"
COLLECTION  = "text_search"
MODEL_NAME  = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
DIM         = 384
TOP_K       = 3

DOCS = [
    "Milvus is a vector database built for AI applications.",
    "Docker Compose makes it easy to run multi-container applications.",
    "Retrieval-augmented generation combines search with language models.",
    "Kubernetes orchestrates containerized workloads at scale.",
    "Embeddings represent text as dense numerical vectors.",
    "HNSW is an approximate nearest neighbor search algorithm.",
    "Python is a popular language for machine learning.",
    "Semantic search understands the meaning behind a query.",
    "MinIO is an S3-compatible object storage system.",
    "etcd is a distributed key-value store used for service coordination.",
]

QUERIES = [
    "how does vector search work?",
    "what is used for container orchestration?",
    "storing data in object storage",
]


def main():
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = MilvusClient(uri=MILVUS_URI), SentenceTransformer(MODEL_NAME)
    client, model = model

    print(f"Connected to Milvus at {MILVUS_URI}\n")

    # --- create collection ---
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    client.create_collection(collection_name=COLLECTION, dimension=DIM)

    # --- embed and ingest ---
    vectors = model.encode(DOCS, show_progress_bar=False).tolist()
    data = [{"id": i, "vector": vectors[i], "text": DOCS[i]} for i in range(len(DOCS))]
    res = client.insert(collection_name=COLLECTION, data=data)
    print(f"Inserted {res['insert_count']} documents\n")

    # --- search ---
    for query in QUERIES:
        query_vec = model.encode([query]).tolist()
        results = client.search(
            collection_name=COLLECTION,
            data=query_vec,
            limit=TOP_K,
            output_fields=["text"],
        )
        print(f"Query: \"{query}\"")
        for hit in results[0]:
            print(f"  [{hit['distance']:.4f}] {hit['entity']['text']}")
        print()

    # --- cleanup ---
    client.drop_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' dropped. Done.")


if __name__ == "__main__":
    main()
