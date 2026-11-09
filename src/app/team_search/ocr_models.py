"""Explicit download of local OCR language files from the Tesseract project."""
from pathlib import Path
import urllib.request
from .config import MODEL_DIR

def download():
    directory=MODEL_DIR.parent/'tessdata'
    directory.mkdir(parents=True,exist_ok=True)
    for language in ('rus','eng'):
        path=directory/(language+'.traineddata')
        if path.is_file():
            continue
        temporary=path.with_suffix('.download')
        urllib.request.urlretrieve(
            f'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{language}.traineddata', temporary)
        if temporary.stat().st_size < 100000:
            raise ValueError('Файл OCR не загружен полностью')
        temporary.replace(path)
    print(f'OCR rus+eng: {directory}')
