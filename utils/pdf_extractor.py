# Filename: utils/pdf_extractor.py (FINAL UPGRADED VERSION)

import fitz # PyMuPDF
import re
from typing import List, Dict, Any
from langdetect import detect, lang_detect_exception

# This is necessary for consistent results with langdetect
from langdetect import DetectorFactory
DetectorFactory.seed = 0

class PdfExtractor:
    """
    Extracts enriched blocks from a PDF. This final version uses:
    1. Language-agnostic font flags for reliable style detection (bold/italic).
    2. LangDetect to identify the language of each text block.
    """
    def __init__(self):
        # No models to load here.
        pass

    def extract_enriched_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        doc = fitz.open(pdf_path)
        all_blocks_data = []
        for page_num, page in enumerate(doc, 1):
            # Since YOLO is removed, we no longer detect tables here.
            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            sorted_blocks = sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))
            for block in sorted_blocks:
                if block.get('type') == 0 and block.get('lines'):
                    processed_block = self._process_block(block, page, page_num)
                    if processed_block and processed_block['text']:
                        all_blocks_data.append(processed_block)
        return self._post_process_spacing(all_blocks_data)

    def _process_block(self, block: Dict, page: fitz.Page, page_num: int) -> Dict:
        block_bbox = fitz.Rect(block['bbox'])
        text_parts, font_sizes, font_names, span_flags = [], [], [], []
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text_parts.append(span['text'])
                font_sizes.append(span['size'])
                font_names.append(span['font'])
                span_flags.append(span['flags'])

        if not text_parts: return None
        block_text = re.sub(r'\s+', ' ', " ".join(text_parts)).strip()
        
        # ## NEW ##: Language Detection
        detected_lang = 'unknown'
        if block_text:
            try:
                detected_lang = detect(block_text)
            except lang_detect_exception.LangDetectException:
                detected_lang = 'unknown'

        # ## NEW ##: Language-agnostic style detection using PyMuPDF's font flags
        is_bold = sum(1 for flags in span_flags if flags & 2**4) > len(span_flags) / 2
        is_italic = sum(1 for flags in span_flags if flags & 2**1) > len(span_flags) / 2
        
        most_common_font = max(set(font_names), key=font_names.count) if font_names else "N/A"
        
        # Since YOLO is removed, we can add a simple rule for table detection if needed,
        # or just assume False. For simplicity, we will assume False.
        is_in_table = False 
        column = 1 if block_bbox.x0 < page.rect.width / 2 else 2

        return {
            "text": block_text,
            "language": detected_lang, # Added language
            "bbox": {'x0': block_bbox.x0, 'y0': block_bbox.y0, 'x1': block_bbox.x1, 'y1': block_bbox.y1},
            "page_number": page_num,
            "page_width": page.rect.width,
            "page_height": page.rect.height,
            "font_size": round(max(set(font_sizes), key=font_sizes.count), 2) if font_sizes else 0,
            "font_name": most_common_font,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "char_count": len(block_text), # Changed from word_count to char_count
            "line_count": len(block.get('lines', [])),
            "is_in_table": is_in_table,
            "column": column
        }

    def _post_process_spacing(self, all_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # This function remains unchanged.
        for i, current_block in enumerate(all_blocks):
            if i + 1 < len(all_blocks) and all_blocks[i+1]['page_number'] == current_block['page_number']:
                current_block['vertical_space_after'] = all_blocks[i+1]['bbox']['y0'] - current_block['bbox']['y1']
            else:
                current_block['vertical_space_after'] = 100
            if i > 0 and all_blocks[i-1]['page_number'] == current_block['page_number']:
                 current_block['vertical_space_before'] = current_block['bbox']['y0'] - all_blocks[i-1]['bbox']['y1']
            else:
                 current_block['vertical_space_before'] = 100
        return all_blocks