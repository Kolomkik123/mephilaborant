"""One model, metric and project-relative storage for build and search."""
import os
from pathlib import Path
from functools import lru_cache
from app.config import ROOT, settings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METRIC = "cosine"
INDEX_DIR = ROOT / os.getenv("TEAM_INDEX_DIR", "data/team_index")
PARSED_DIR = ROOT / os.getenv("TEAM_PARSED_DIR", "data/team_parsed")
MODEL_DIR = ROOT / os.getenv("TEAM_MODEL_DIR", "models/all-MiniLM-L6-v2")
RAW_DIR = ROOT / os.getenv("TEAM_RAW_DIR", str(settings.methodics_dir))
MANIFEST = INDEX_DIR / "manifest.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

@lru_cache(maxsize=2)
def embedding_function(allow_download=False):
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    required = ("model.onnx", "tokenizer.json", "config.json", "vocab.txt",
                "special_tokens_map.json", "tokenizer_config.json")
    if not allow_download and not all((MODEL_DIR/"onnx"/name).is_file() for name in required):
        raise ValueError("Нет локальной модели поиска. Выполните: python run.py search-model")
    model = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    model.DOWNLOAD_PATH = MODEL_DIR
    return model

def client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(INDEX_DIR), settings=Settings(anonymized_telemetry=False))
