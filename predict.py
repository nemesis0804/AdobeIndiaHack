# Filename: predict.py (UPGRADED with Advanced Heuristics)

import os
import json
import argparse
import joblib
import numpy as np
import re
from typing import List, Dict, Any
from utils.pdf_extractor import PdfExtractor
from utils.feature_extractor_lite import FeatureExtractorLite

def apply_advanced_rules(block: Dict[str, Any], predicted_label: str, median_font: float) -> str:
    """
    Applies a much more robust set of rules to eliminate common false positives
    like TOC entries, footers, and table headers.
    """
    text = block['text'].strip()
    if not text:
        return 'NONE'

    # --- STRONG NEGATIVE FILTERS (Early Exit) ---
    # These patterns are almost never headings.

    # 1. Filter out Table of Contents (TOC) entries.
    # Pattern: text ending with ...... 123 or a page number.
    if re.search(r'\.{3,}\s*\d+\s*$', text):
        return 'NONE'
    # Pattern: A heading candidate followed by just a number at the end of the line.
    if len(text.split()) > 2 and re.search(r'\s+\d+\s*$', text):
         # Exclude if it's a genuine numbered heading like "1. Introduction"
        if not re.match(r'^\d+\.\s+', text):
            return 'NONE'

    # 2. Filter out footer/header text.
    # Text at the very top or bottom of the page with a small font is likely a header/footer.
    y_pos_normalized = block['bbox']['y1'] / block.get('page_height', 792)
    if (y_pos_normalized > 0.9 or y_pos_normalized < 0.1) and block['font_size'] <= median_font:
        return 'NONE'

    # 3. Filter out table headers or other non-heading metadata.
    # Example: "Version Date Remarks". More than 2 words, title-cased, but not a sentence.
    words = text.split()
    if len(words) > 2 and all(word.istitle() or word.isupper() for word in words) and block['font_size'] < median_font * 1.1:
         # This is likely a table header, not a section heading.
         return 'NONE'
    
    # 4. Filter out simple version numbers or dates on the title page.
    if block['page_number'] == 1 and re.fullmatch(r'Version\s?[\d\.]+|[\d\w\s,]+20\d{2}', text, re.IGNORECASE):
        return 'NONE'


    # --- HEADING SCORING SYSTEM (from previous version, but now applied to filtered blocks) ---
    heading_score = 0
    font_size = block.get('font_size', 0)
    is_bold = block.get('is_bold', False)
    space_before = block.get('vertical_space_before', 0)
    relative_size = font_size / median_font if median_font > 0 else 1.0

    if is_bold: heading_score += 2
    if relative_size > 1.25: heading_score += 3
    elif relative_size > 1.15: heading_score += 2
    if space_before > font_size * 1.5: heading_score += 2
    if len(text) > 150: heading_score -= 3
    if text.endswith(('.', ':', ';')): heading_score -= 2

    # --- DECISION LOGIC ---
    if predicted_label.startswith('H') and heading_score < 1:
        return 'NONE'
    if predicted_label == 'NONE' and heading_score >= 4:
        if re.match(r'^[1-9]\d*\.[1-9]\d*\.\s+', text): return 'H2'
        if re.match(r'^[1-9]\d*\.\s+', text): return 'H1'
        return 'H1'

    # Rule: A TITLE can ONLY be on page 1.
    if predicted_label == 'TITLE' and block['page_number'] != 1:
        return 'H1'

    return predicted_label

def structure_final_output(labeled_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Formats the flat list of labeled blocks into the desired nested JSON structure.
    Includes logic to merge multi-line titles.
    """
    title = "Untitled Document"
    outline = []
    title_block_indices = []

    # Heuristic to find the title: Find the block with the largest font on page 1.
    page1_blocks = [b for b in labeled_blocks if b.get('page_number') == 1]
    
    if page1_blocks:
        def sort_key(b):
            text = b[0]['text'].strip() # b is now (block, index)
            if re.fullmatch(r'[\d\W\s]+', text) or len(text) < 4 or text.lower() == 'copyright':
                return 0
            return b[0].get('font_size', 0)

        # Keep track of original index
        page1_blocks_with_indices = [(block, i) for i, block in enumerate(labeled_blocks) if block.get('page_number') == 1]
        
        if page1_blocks_with_indices:
            page1_blocks_with_indices.sort(key=sort_key, reverse=True)
            
            # Start with the highest-scored block as the first line of the title
            title_block, title_start_index = page1_blocks_with_indices[0]
            title_parts = [title_block['text']]
            title_block_indices.append(title_start_index)
            
            # ** NEW: Attempt to merge subsequent lines into the title **
            last_title_block = title_block
            for i in range(page1_blocks_with_indices.index((title_block, title_start_index)) + 1, len(page1_blocks_with_indices)):
                next_block, next_block_index = page1_blocks_with_indices[i]
                
                # Condition: is the next block stylistically similar and vertically close?
                is_similar_font = abs(next_block['font_size'] - last_title_block['font_size']) < 1.0
                is_close_vertically = (next_block['bbox']['y0'] - last_title_block['bbox']['y1']) < last_title_block['font_size']
                
                if is_similar_font and is_close_vertically:
                    title_parts.append(next_block['text'])
                    title_block_indices.append(next_block_index)
                    last_title_block = next_block
                else:
                    # The title block sequence is broken
                    break
            
            title = " ".join(title_parts)


    # Build the outline from all blocks that are headings AND were not used in the title
    for i, block in enumerate(labeled_blocks):
        # We check the original index `i`
        if i in title_block_indices:
            continue

        label = block.get('final_label')
        if label and label.startswith('H'):
            outline.append({
                "level": label,
                "text": block['text'],
                "page": block['page_number']
            })

    return {"title": title.strip(), "outline": outline}


def predict_outline(pdf_path, model_path, output_path=None):
    """
    Loads a model, predicts labels sequentially, applies advanced rules, and structures the output.
    """
    print(f"--- Running Prediction Phase (Hybrid ML + Advanced Heuristics) ---")
    
    # Steps 1 & 2: Load Model and Extract Blocks (Unchanged)
    print(f"Loading model bundle from: {model_path}...")
    try:
        model_bundle = joblib.load(model_path)
        model = model_bundle['model']
        inverse_label_map = {v: k for k, v in model_bundle['label_mapping'].items()}
    except Exception as e:
        print(f"FATAL ERROR: Could not load model bundle. Error: {e}")
        return

    print(f"Extracting and processing blocks from: {pdf_path}...")
    pdf_extractor = PdfExtractor()
    blocks = pdf_extractor.extract_enriched_blocks(pdf_path)
    if not blocks:
        print("Warning: No text blocks were found.")
        return

    # Step 3: Sequential Prediction with Advanced Rules (Main Change Here)
    print("Generating features and predicting labels with advanced rules...")
    feature_extractor = FeatureExtractorLite()
    
    font_map = feature_extractor._get_font_mapping(blocks)
    lang_map = feature_extractor._get_language_mapping(blocks)
    doc_font_sizes = [b.get('font_size', 0) for b in blocks if b.get('font_size', 0) > 6]
    median_font = np.median(doc_font_sizes) if doc_font_sizes else 12.0

    last_heading_info = {'index': -1, 'level': 0, 'font_size': median_font}
    all_labeled_blocks = []

    for i, block in enumerate(blocks):
        current_features_list = feature_extractor._get_block_features(
            block, i, median_font, font_map, lang_map, last_heading_info
        )
        current_features_np = np.array([current_features_list], dtype=np.float32)

        predicted_label_int = model.predict(current_features_np)[0]
        initial_label = inverse_label_map.get(predicted_label_int, 'NONE')
        
        # Apply the NEW, more advanced rules
        final_label = apply_advanced_rules(block, initial_label, median_font)
        
        block['final_label'] = final_label
        all_labeled_blocks.append(block)

        if final_label.startswith('H'):
            try:
                level = int(final_label[1:])
                last_heading_info = {'index': i, 'level': level, 'font_size': block.get('font_size', median_font)}
            except (ValueError, IndexError):
                pass
        elif final_label == 'TITLE':
            last_heading_info = {'index': i, 'level': 0, 'font_size': block.get('font_size', median_font)}

    # Steps 4 & 5: Structure Output and Save (Unchanged, but uses new structuring function)
    print("Structuring the final outline...")
    final_output = structure_final_output(all_labeled_blocks)
    output_json_string = json.dumps(final_output, indent=4, ensure_ascii=False)

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_json_string)
        print(f"\n✅ Structured outline saved successfully to: {output_path}")
    else:
        print("\n--- Generated Structured JSON Outline ---")
        print(output_json_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict PDF outline using a trained model and advanced heuristic rules.")
    parser.add_argument('--input', required=True, help="Path to the new PDF file to process.")
    parser.add_argument('--model', required=True, help="Path to the saved model bundle (.joblib).")
    parser.add_argument('--output', help="Optional. Path to save the output as a JSON file.")
    
    args = parser.parse_args()
    
    predict_outline(args.input, args.model, args.output)