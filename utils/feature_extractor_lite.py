# Filename: utils/feature_extractor_lite.py (SPEED-FOCUSED VERSION)

import re
import numpy as np
from typing import List, Dict, Any, Tuple

class FeatureExtractorLite:
    def __init__(self):
        """
        A lightweight feature extractor that does NOT use heavy sentence transformer models.
        It relies on fast-to-compute typographical, positional, and simple content features.
        """
        # ## MODIFIED ##: Removed all embedding features.
        self.feature_names = [
            'font_size', 'is_bold', 'is_italic', 'relative_font_size', 'font_name_id', 'bold_x_rel_size',
            'line_width_ratio', 'y_position_normalized', 'x_position_normalized', 'is_centered',
            'is_in_table', 'column', 'space_before_ratio', 'space_after_ratio',
            'line_count', 'char_count', 'ends_with_punct', 'language_id',
            'last_heading_level', 'distance_from_last_heading', 'font_size_vs_last_heading'
        ]

    def _get_font_mapping(self, all_blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        unique_fonts = sorted(list(set(b.get('font_name', 'default') for b in all_blocks)))
        return {font_name: i for i, font_name in enumerate(unique_fonts)}

    def _get_language_mapping(self, all_blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        unique_langs = sorted(list(set(b.get('language', 'unknown') for b in all_blocks)))
        return {lang_name: i for i, lang_name in enumerate(unique_langs)}

    def extract_features(self, all_blocks: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, int]]:
        if not all_blocks: return np.array([]), {}
        
        # No more slow embedding generation!
        
        doc_font_sizes = [b.get('font_size', 0) for b in all_blocks if b.get('font_size', 0) > 6]
        median_font = np.median(doc_font_sizes) if doc_font_sizes else 12.0
        font_map = self._get_font_mapping(all_blocks)
        lang_map = self._get_language_mapping(all_blocks)
        
        heading_context = []
        last_heading_info = {'index': -1, 'level': 0, 'font_size': median_font}
        
        for i, block in enumerate(all_blocks):
            heading_context.append(last_heading_info.copy())
            label = block.get('label', 'NONE')
            if label.startswith('H'):
                try:
                    level = int(label[1:])
                    last_heading_info = { 'index': i, 'level': level, 'font_size': block.get('font_size', median_font) }
                except (ValueError, IndexError): pass
        
        features_matrix = [
            self._get_block_features(block, i, median_font, font_map, lang_map, heading_context[i])
            for i, block in enumerate(all_blocks)
        ]
        
        # The returned lang_map isn't strictly needed for the bundle, but we keep it here.
        return np.array(features_matrix, dtype=np.float32), font_map

    def _get_block_features(self, block: Dict[str, Any], index: int, median_font: float, font_map: Dict[str, int], lang_map: Dict[str, int], context: Dict) -> List[float]:
        text = block.get('text', '')
        font_size = block.get('font_size', 0)
        is_bold = float(block.get('is_bold', False))
        is_italic = float(block.get('is_italic', False))
        font_name = block.get('font_name', 'default')
        bbox = block.get('bbox', {'x0': 0, 'y0': 0, 'x1': 0, 'y1': 0})
        page_width = block.get('page_width', 612.0)
        relative_size = font_size / median_font if median_font > 0 else 1.0
        last_heading_level = float(context['level'])
        distance_from_last_heading = float(index - context['index']) if context['index'] != -1 else 100.0
        last_heading_font_size = context['font_size'] if context['font_size'] > 0 else median_font
        font_size_vs_last_heading = font_size / last_heading_font_size if last_heading_font_size > 0 else 1.0
        language_code = block.get('language', 'unknown')
        language_id = float(lang_map.get(language_code, -1.0))

        # This feature list contains NO slow embedding features.
        manual_features = [
            font_size, is_bold, is_italic, relative_size, float(font_map.get(font_name, -1)), is_bold * relative_size,
            (bbox['x1'] - bbox['x0']) / page_width if page_width > 0 else 0,
            bbox['y0'] / block.get('page_height', 792.0) if block.get('page_height', 792.0) > 0 else 0,
            bbox['x0'] / page_width if page_width > 0 else 0,
            float(abs(((bbox['x0'] + bbox['x1']) / 2) / page_width - 0.5) < 0.15) if page_width > 0 else 0,
            float(block.get('is_in_table', False)), float(block.get('column', 1)),
            block.get('vertical_space_before', 50.0) / font_size if font_size > 0 else 5.0,
            block.get('vertical_space_after', 50.0) / font_size if font_size > 0 else 5.0,
            float(block.get('line_count', 1)), float(block.get('char_count', 0)),
            float(text.strip().endswith((':', '.', '。', '：', '!', '?'))), language_id,
            last_heading_level, distance_from_last_heading, font_size_vs_last_heading
        ]
        
        return manual_features