# Milvus Notes

A local Milvus development environment with sample clients for vector ingestion and retrieval.

## Repository Structure

```
milvus-notes/
├── docker/                  # Docker Compose setup (local dev)
│   ├── docker-compose.yml
│   └── configs/
│       └── milvus.yaml      # local filesystem storage override (no MinIO)
├── kubernetes/              # Helm / kind setup (coming soon)
└── clients/
    └── python/
        ├── basic_demo.py    # smoke test: random vectors, full lifecycle
        ├── text_search.py   # semantic search with sentence-transformers
        └── pyproject.toml
```

## Docker Setup

### Prerequisites

- [Podman](https://podman.io/) (or Docker)
- Podman machine running: `podman machine start`

### Start

```bash
cd docker
podman compose up -d
# or: docker compose up -d
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| Milvus gRPC | `19530` | Main endpoint — clients connect here |
| Milvus REST | `9091` | HTTP API |
| Attu UI | `8000` | Web UI → http://localhost:8000 |

### Notes

- **No MinIO required.** The `configs/milvus.yaml` override sets `storageType: local`, so Milvus writes vector data directly to a named volume instead of object storage. This keeps the dev stack to 2 containers (etcd + milvus) plus the optional Attu UI.
- etcd is still required for Milvus metadata and coordination.
- Data persists across restarts via Docker/Podman named volumes (`etcd_data`, `milvus_data`).

### Stop

```bash
cd docker
podman compose down
```

To also wipe all data:

```bash
podman compose down -v
```

## Python Clients

Requires Python 3.9+ and [uv](https://docs.astral.sh/uv/). Start the Docker stack first.

```bash
cd clients/python
uv sync
```

| Script | What it does |
|--------|-------------|
| `basic_demo.py` | Smoke test — inserts 1000 random vectors and searches. No ML model needed. |
| `text_search.py` | Embeds real text with `all-MiniLM-L6-v2`, inserts, then searches by query string. First run downloads the model (~90 MB). |

```bash
uv run python basic_demo.py
uv run python text_search.py
```

## Kubernetes

See [`kubernetes/`](./kubernetes/) for a Helm-based setup targeting a local [kind](https://kind.sigs.k8s.io/) cluster.
