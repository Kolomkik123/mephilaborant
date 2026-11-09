from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel
from typing import Optional, List

# Модель ДОЛЖНА совпадать с той, что в build_index.py (all-MiniLM-L6-v2)
print("Загрузка модели для поиска...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(persist_directory="data/index", embedding_function=embeddings)

class Source(BaseModel):
    text: str
    work: str
    page: int
    similarity: float

def search_manuals(query: str, work: Optional[str] = None, k: int = 5) -> List[Source]:
    """
    Поиск по методичкам.
    :param query: Текст запроса
    :param work: Номер лабораторной работы (например, "16"), если нужен фильтр
    :param k: Количество возвращаемых чанков
    """
    
    where_filter = {"work": work} if work else None
    
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=where_filter
    )
    
    sources = []
    # Порог близости (настройте, если поиск выдает ерунду)
    SIMILARITY_THRESHOLD = 0.3 
    
    for doc, score in results:
        # ChromaDB с косинусным расстоянием возвращает distance (0=идеально)
        similarity = 1 - score 
        
        if similarity >= SIMILARITY_THRESHOLD:
            sources.append(Source(
                text=doc.page_content,
                work=doc.metadata.get("work", "Unknown"),
                page=doc.metadata.get("page", 0),
                similarity=round(similarity, 3)
            ))
            
    return sources