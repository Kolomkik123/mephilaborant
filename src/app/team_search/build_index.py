"""Colleague's page chunking -> MiniLM -> Chroma pipeline.

Build a new versioned collection, then switch a small manifest atomically.
Neither the calculator's old index nor the previous collection is deleted.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from . import config

def split_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Некорректные размер фрагмента и перекрытие")
    start=0
    while start<len(text):
        end=min(start+size,len(text))
        if end<len(text):
            cut=max(text.rfind("\n",start+size//2,end),text.rfind(". ",start+size//2,end))
            if cut>start+overlap:
                end=cut+1
        chunk=text[start:end].strip()
        if chunk:
            yield chunk
        if end==len(text):
            break
        start=end-overlap

def prepare_documents(parsed_files):
    documents=[]
    from .parse_manuals import normalize_work
    for file_path in parsed_files:
        path=Path(file_path)
        pages=json.loads(path.read_text(encoding="utf-8"))
        for record in pages:
            source=record.get("source_file",path.stem+".pdf")
            for index,chunk in enumerate(split_text(record["text"])):
                identity=f"{source}|{record['page']}|{record['work']}|{index}|{chunk}"
                documents.append({"id":hashlib.sha256(identity.encode()).hexdigest(),"text":chunk,
                    "meta":{"work":normalize_work(record["work"]),"page":int(record["page"]),
                            "source_file":source,"section":"unknown","chunk_number":index}})
    if not documents:
        raise ValueError("Нет текстовых фрагментов для индекса")
    return documents

def build_database(parsed_files=None):
    paths=list(parsed_files) if parsed_files is not None else sorted(config.PARSED_DIR.glob("*.json"))
    documents=prepare_documents(paths)
    fingerprint=hashlib.sha256(json.dumps({'documents':documents,'model':config.MODEL_NAME,
        'metric':config.METRIC,'runtime':'chroma-onnx'},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if config.MANIFEST.is_file():
        previous=json.loads(config.MANIFEST.read_text(encoding='utf-8'))
        if previous.get('fingerprint')==fingerprint:
            existing=config.client().get_collection(previous['collection'],embedding_function=None)
            if existing.count()==len(documents):
                return {**previous,'reused':True}
    embed=config.embedding_function()
    collection_name="team_manuals_"+uuid.uuid4().hex
    collection=config.client().create_collection(collection_name,embedding_function=None,
        metadata={"hnsw:space":config.METRIC,"model":config.MODEL_NAME})
    for start in range(0,len(documents),64):
        batch=documents[start:start+64]
        texts=[item["text"] for item in batch]
        collection.add(ids=[item["id"] for item in batch],documents=texts,
            metadatas=[item["meta"] for item in batch],embeddings=embed(texts))
    manifest={"collection":collection_name,"model":config.MODEL_NAME,"metric":config.METRIC,"fingerprint":fingerprint,
        "embedding_runtime":"chroma-onnx","chunk_size":config.CHUNK_SIZE,"chunk_overlap":config.CHUNK_OVERLAP,
        "fragments":len(documents),"documents":len({item["meta"]["source_file"] for item in documents}),
        "sources":sorted({item["meta"]["source_file"] for item in documents}),
        "built_utc":datetime.now(timezone.utc).isoformat()}
    config.INDEX_DIR.mkdir(parents=True,exist_ok=True)
    temporary=config.MANIFEST.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(config.MANIFEST)
    return manifest
