"""Adaptation of colleague's search_manuals(query, work, k) interface.

The same MiniLM model is run by Chroma's ONNX implementation; this avoids
installing PyTorch/LangChain alongside the calculator. Imports do not load
a model, create an index, or contact the network.
"""
import json
from pathlib import Path
from pydantic import BaseModel
from . import config
from .parse_manuals import normalize_work

class Source(BaseModel):
    text: str
    work: str
    page: int
    similarity: float
    source_file: str = ""
    chunk_id: str = ""
    chunk_number: int = 0
    match_type: str = "semantic"

def index_manifest():
    if not config.MANIFEST.is_file():
        return {"documents":0,"fragments":0,"sources":[]}
    return json.loads(config.MANIFEST.read_text(encoding="utf-8"))

def collection():
    manifest=index_manifest()
    if not manifest.get("collection"):
        raise ValueError("Индекс поисковика команды не собран. Выполните: python run.py search-index")
    if (manifest.get("model"),manifest.get("metric"),manifest.get("embedding_runtime")) != (config.MODEL_NAME,config.METRIC,"chroma-onnx"):
        raise ValueError("Модель или метрика индекса отличается от поиска. Пересоберите: python run.py search-index")
    return config.client().get_collection(manifest["collection"],embedding_function=None)

def search_manuals(query: str, work: str | None = None, k: int = 5) -> list[Source]:
    if not query.strip() or k<1:
        return []
    store=collection()
    if not store.count():
        return []
    result=store.query(query_embeddings=config.embedding_function()([query]),
        n_results=min(k,store.count()),where={"work":normalize_work(work)} if work else None,
        include=["documents","metadatas","distances"])
    sources=[]
    for text,metadata,distance,chunk_id in zip(result["documents"][0],result["metadatas"][0],result["distances"][0],result["ids"][0]):
        similarity=1-float(distance)  # index metric is explicitly cosine
        if similarity>=0.3:
            sources.append(Source(text=text,work=metadata["work"],page=metadata["page"],
                similarity=similarity,source_file=metadata["source_file"],chunk_id=chunk_id,
                chunk_number=metadata["chunk_number"]))
    return sources

def fragments_by_work(work: str, limit=5) -> list[Source]:
    # A specified lab is an exact metadata restriction. Do not drop its
    # formula because a generic Russian request is distant from English MiniLM.
    records=collection().get(where={"work":normalize_work(work)},include=["documents","metadatas"])
    sources=[Source(text=text,work=meta["work"],page=meta["page"],similarity=0,match_type="metadata",
                    source_file=meta["source_file"],chunk_id=identifier,chunk_number=meta["chunk_number"])
             for text,meta,identifier in zip(records["documents"],records["metadatas"],records["ids"])]
    sources.sort(key=lambda item:(item.source_file,item.page,item.chunk_number))
    return sources[:limit]
