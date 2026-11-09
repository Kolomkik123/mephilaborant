"""Persistent text cache: real PDF extraction and invalidation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import pymupdf
from app.team_search.parse_manuals import parse_pdf, page_records


class PDFCacheTests(unittest.TestCase):
    def test_spaced_headers_and_contents(self):
        for title in ('Р а б о т а  8', 'Р а б от а  15'):
            records,work=page_records(title+'\nТеория','Введение',4,'manual.pdf')
            self.assertEqual(work,title.split()[-1])
            self.assertEqual(records[0]['work'],work)
        records,work=page_records('СОДЕРЖАНИЕ\nРабота 8.\nТема\nРабота 9.\nТема','15',87,'manual.pdf')
        self.assertEqual(work,'15')
        self.assertEqual([r['work'] for r in records],['Содержание'])
        records,work=page_records('15\nР а б о т а 9\nТеория','8',15,'manual.pdf')
        self.assertEqual([r['work'] for r in records],['9'])

    def test_unchanged_pdf_is_not_opened_again(self):
        sample=Path(__file__).resolve().parents[1]/'data/methodics/ОБРАЗЕЦ_механика_практикум.pdf'
        with TemporaryDirectory() as temp:
            output=parse_pdf(sample,Path(temp))
            before=output.stat().st_mtime_ns
            with patch('app.team_search.parse_manuals.pymupdf.open',side_effect=AssertionError('PDF reopened')):
                self.assertEqual(parse_pdf(sample,Path(temp)),output)
            self.assertEqual(before,output.stat().st_mtime_ns)
            self.assertIn('1.2',{p['work'] for p in json.loads(output.read_text(encoding='utf-8'))})

    def test_changed_pdf_and_corrupt_cache_are_reparsed(self):
        with TemporaryDirectory() as temp:
            path=Path(temp)/'test.pdf'
            def save(text):
                with pymupdf.open() as doc:
                    page=doc.new_page()
                    page.insert_text((50,50),text)
                    doc.save(path)
            save('first version')
            output=parse_pdf(path,Path(temp)/'parsed')
            save('second version')
            parse_pdf(path,output.parent)
            self.assertIn('second version',output.read_text(encoding='utf-8'))
            output.write_text('broken',encoding='utf-8')
            parse_pdf(path,output.parent)
            self.assertIn('second version',output.read_text(encoding='utf-8'))

    def test_multiple_dotted_work_headers(self):
        records,work=page_records('Работа №1.2\nПервая\nЛабораторная работа 2.3\nВторая','',3,'a.pdf')
        self.assertEqual([p['work'] for p in records],['1.2','2.3'])
        self.assertEqual(work,'2.3')

    def test_ocr_missing_models_does_not_write_empty_success(self):
        sample=Path(__file__).resolve().parents[1]/'data/methodics/ОБРАЗЕЦ_механика_практикум.pdf'
        with TemporaryDirectory() as temp:
            path=Path(temp)/'scan.pdf'
            with pymupdf.open(sample) as source:
                pixmap=source[5].get_pixmap(dpi=100)
                with pymupdf.open() as scan:
                    page=scan.new_page(width=source[5].rect.width,height=source[5].rect.height)
                    page.insert_image(page.rect,stream=pixmap.tobytes('png'))
                    scan.save(path)
            with patch.dict('os.environ',{'OCR_TESSDATA':str(Path(temp)/'absent')}):
                with self.assertRaisesRegex(ValueError,'OCR'):
                    parse_pdf(path,Path(temp)/'parsed')
            self.assertFalse(list((Path(temp)/'parsed').glob('*.json')))
