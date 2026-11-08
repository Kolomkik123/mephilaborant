import pymupdf
import pymupdf4llm
import os
import json
import re

def parse_pdf(pdf_path, output_dir="data/parsed"):
    os.makedirs(output_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    parsed_pages = []
    
    work_pattern = re.compile(r'Лабораторная работа\s*(\d+[а-я]?)', re.IGNORECASE)
    current_work = "Введение"

    # pymupdf4llm умеет обрабатывать весь документ целиком и отдавать markdown постранично
    md_text = pymupdf4llm.to_markdown(doc, page_chunks=True)
    
    for i, page_data in enumerate(md_text):
        text = page_data["text"]
        
        match = work_pattern.search(text)
        if match:
            current_work = match.group(1)
            
        if text.strip():
            parsed_pages.append({
                "work": current_work,
                "page": i + 1,
                "text": text
            })

    filename = os.path.basename(pdf_path).replace(".pdf", ".json")
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_pages, f, ensure_ascii=False, indent=2)
        
    print(f"Файл {pdf_path} успешно спарсен. Сохранено {len(parsed_pages)} страниц.")

if __name__ == "__main__":
    raw_dir = "data/raw"
    for file in os.listdir(raw_dir):
        if file.endswith(".pdf"):
            parse_pdf(os.path.join(raw_dir, file))