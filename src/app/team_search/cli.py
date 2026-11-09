"""Explicit model download, indexing and real integration smoke checks."""
import json

def execute(command):
    if command=='search-ocr-model':
        from .ocr_models import download
        download()
    elif command=="search-model":
        from .config import embedding_function, MODEL_DIR
        embed=embedding_function(allow_download=True)
        vectors=embed(["Model readiness check"])
        print(f"MiniLM ready: {MODEL_DIR}; dimensions={len(vectors[0])}")
    elif command=="search-parse":
        from .parse_manuals import parse_all
        print("\n".join(str(path) for path in parse_all()))
    elif command=="search-index":
        from .parse_manuals import parse_all
        from .build_index import build_database
        print(json.dumps(build_database(parse_all()),ensure_ascii=False,indent=2))
    else:
        import os
        os.environ['LLM_PROVIDER']='mock'
        from .verify import verify
        return verify()
    return 0
