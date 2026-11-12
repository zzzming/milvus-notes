"""
Basic Milvus demo: full lifecycle with random vectors.
No embedding model needed — good as a connectivity/smoke test.

Usage: python basic_demo.py
"""

import random
from pymilvus import MilvusClient

MILVUS_URI = "http://localhost:19530"
COLLECTION  = "basic_demo"
DIM         = 128
NUM_VECTORS = 1000
TOP_K       = 5


def main():
    client = MilvusClient(uri=MILVUS_URI)
    print(f"Connected to Milvus at {MILVUS_URI}")

    # --- create collection ---
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
    client.create_collection(collection_name=COLLECTION, dimension=DIM)
    print(f"Collection '{COLLECTION}' created (dim={DIM})")

    # --- ingest ---
    data = [
        {"id": i, "vector": [random.random() for _ in range(DIM)], "label": f"item_{i}"}
        for i in range(NUM_VECTORS)
    ]
    res = client.insert(collection_name=COLLECTION, data=data)
    print(f"Inserted {res['insert_count']} vectors")

    # --- search ---
    query_vec = [random.random() for _ in range(DIM)]
    results = client.search(
        collection_name=COLLECTION,
        data=[query_vec],
        limit=TOP_K,
        output_fields=["label"],
    )

    print(f"\nTop {TOP_K} nearest neighbors:")
    for hit in results[0]:
        print(f"  id={hit['id']:<6} distance={hit['distance']:.4f}  label={hit['entity']['label']}")

    # --- cleanup ---
    client.drop_collection(COLLECTION)
    print(f"\nCollection '{COLLECTION}' dropped. Done.")


if __name__ == "__main__":
    main()
