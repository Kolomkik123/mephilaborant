"""PDF -> per-page records, based on colleague's parse_manuals.py.

PyMuPDF is already installed in the calculator. No extra parser or network
is required. Markdown extraction can be added separately for complex PDFs.
"""
import json
import re
import hashlib
import os
import uuid
from pathlib import Path
import pymupdf
from .config import RAW_DIR, PARSED_DIR, MODEL_DIR

WORK_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:лабораторная\s+)?р[ \t]*а[ \t]*б[ \t]*о[ \t]*т[ \t]*а\s*(?:№|No\.?|N)?\s*"
    r"(\d+(?:[.\-]\d+)*(?:[а-яё])?)(?!\d)", re.I | re.M)

def normalize_work(work):
    return str(work).strip().lower().replace("-", ".")

def page_records(text, current_work, page, source_file):
    # Contents entries are references, not starts of laboratory chapters.
    if re.search(r'^\s*(?:содержание|оглавление)\s*$', text, re.I | re.M):
        return [{'work':'Содержание','page':page,'text':text.strip(),'source_file':source_file}],current_work
    matches = list(WORK_PATTERN.finditer(text))
    records = []
    position = 0
    for match in matches:
        prefix = text[position:match.start()].strip()
        if prefix and not prefix.isdecimal():
            records.append({"work":current_work,"page":page,"text":prefix,"source_file":source_file})
        current_work = normalize_work(match.group(1))
        position = match.start()
    tail = text[position:].strip()
    if tail:
        records.append({"work":current_work,"page":page,"text":tail,"source_file":source_file})
    return records,current_work

def parse_pdf(pdf_path, output_dir=PARSED_DIR):
    path=Path(pdf_path)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(path.name.encode()).hexdigest()[:12]
    target=output_dir/(path.stem+'_'+key+'.json')
    metadata=target.with_suffix('.meta')
    tessdata=Path(os.getenv('OCR_TESSDATA',str(MODEL_DIR.parent/'tessdata')))
    fingerprint={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'parser_version':3,'ocr_languages':'rus+eng','source_file':path.name,
                 'ocr_models':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(tessdata.glob('*.traineddata'))}}
    if target.is_file() and metadata.is_file():
        try:
            previous=json.loads(metadata.read_text(encoding='utf-8'))
            cached=target.read_bytes()
            if (previous.get('fingerprint')==fingerprint and
                previous.get('text_sha256')==hashlib.sha256(cached).hexdigest()):
                return target
        except (ValueError,OSError):
            pass
    records=[]
    current_work="Введение"
    methods=[]
    with pymupdf.open(path) as document:
        for index,page in enumerate(document):
            text=page.get_text('text')
            method='text'
            if page.get_images() or (not text.strip() and page.get_drawings()):
                if not all((tessdata/(lang+'.traineddata')).is_file() for lang in ('rus','eng')):
                    raise ValueError(f'{path.name}, с. {index+1}: нужно OCR. Выполните python run.py search-ocr-model')
                try:
                    textpage=page.get_textpage_ocr(language='rus+eng',dpi=200,
                        full=not bool(text.strip()),tessdata=str(tessdata))
                    text=page.get_text('text',textpage=textpage)
                    method='ocr'
                except Exception as exc:
                    raise ValueError(f'{path.name}, с. {index+1}: ошибка OCR ({type(exc).__name__}). Старый индекс сохранён.') from exc
            methods.append({'page':index+1,'method':method,'characters':len(text)})
            items,current_work=page_records(text,current_work,index+1,path.name)
            records.extend(items)
    if not records:
        raise ValueError(f"В {path.name} нет текстового слоя. Нужен PDF с текстом или отдельное OCR.")
    text=json.dumps(records,ensure_ascii=False,indent=2).encode('utf-8')
    temporary=target.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_bytes(text)
    temporary.replace(target)
    info={'fingerprint':fingerprint,'pages':methods,'text_sha256':hashlib.sha256(text).hexdigest()}
    temporary=metadata.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(metadata)
    return target

def parse_all(raw_dir=RAW_DIR, output_dir=PARSED_DIR):
    paths=sorted(p for p in Path(raw_dir).glob("*") if p.suffix.lower()==".pdf")
    if not paths:
        raise ValueError(f"Не найдены PDF в {raw_dir}")
    return [parse_pdf(path,output_dir) for path in paths]
