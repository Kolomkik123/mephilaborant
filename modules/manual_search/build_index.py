import os
import json
import glob
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

print("Загрузка модели эмбеддингов...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len,
    is_separator_regex=False,
)

def build_database():
    all_documents = []
    parsed_files = glob.glob("data/parsed/*.json")
    
    if not parsed_files:
        print("Ошибка: Нет спарсенных файлов в data/parsed/")
        return

    for file_path in parsed_files:
        with open(file_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
            for page in pages:
                chunks = text_splitter.split_text(page["text"])
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "work": str(page["work"]),
                            "page": page["page"],
                            "source_file": os.path.basename(file_path),
                            "section": "unknown", 
                            "chunk_id": f"{os.path.basename(file_path)}_p{page['page']}_c{i}"
                        }
                    )
                    all_documents.append(doc)

    print(f"Всего создано чанков: {len(all_documents)}")
    
    # Очищаем старую базу
    if os.path.exists("data/index"):
        shutil.rmtree("data/index")
        
    print("Создание пустой базы ChromaDB...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory="data/index"
    )
    
    print("Добавление документов батчами (по 500 шт)...")
    batch_size = 500
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Добавлено {i + len(batch)} из {len(all_documents)}...")
        
    print("Индекс успешно построен и сохранен в data/index/")

if __name__ == "__main__":
    build_database()