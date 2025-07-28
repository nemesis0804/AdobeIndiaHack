# Filename: main.py (FINAL VERSION - FOR TRAINING THE LITE MODEL)

import os
import json
import argparse
import numpy as np
import xgboost as xgb
import joblib
from utils.pdf_extractor import PdfExtractor
# ## CRITICAL CHANGE ##: Import the new, fast "Lite" feature extractor
from utils.feature_extractor_lite import FeatureExtractorLite

# --- Define Label Mapping (The ground truth for your categories) ---
LABEL_MAPPING = {'NONE': 0, 'TITLE': 1, 'H1': 2, 'H2': 3, 'H3': 4, 'H4': 5}

def run_extraction(pdf_path, output_path):
    """
    Phase 1: Extracts enriched block data from a PDF and saves it as a JSON file,
    ready for manual labeling. This function is unchanged.
    """
    print(f"--- Running Phase 1: Extraction for Labeling ---")
    print(f"Processing PDF: {pdf_path}")
    
    # This uses the standard PdfExtractor, which is correct.
    extractor = PdfExtractor()
    enriched_blocks = extractor.extract_enriched_blocks(pdf_path)
    
    # Add a default 'label' key to each block to make labeling easier
    for block in enriched_blocks:
        block['label'] = 'NONE'

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    # Write the JSON file with UTF-8 encoding and ensure_ascii=False
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_blocks, f, indent=2, ensure_ascii=False)
        
    print(f"\nSuccessfully extracted {len(enriched_blocks)} blocks.")
    print(f"Raw data ready for labeling has been saved to: {output_path}")
    print("\nNext Step: Manually review this JSON file and change the 'label' value for each block to the correct category (e.g., 'TITLE', 'H1', 'H2', etc.).")

def run_training(labeled_data_path, model_output_path):
    """
    Phase 2: Trains an XGBoost model on the manually labeled data directory
    using the lightweight FeatureExtractorLite.
    """
    print(f"--- Running Phase 2: Training ---")
    print(f"Loading labeled data from directory: {labeled_data_path}")

    all_blocks = []
    for filename in os.listdir(labeled_data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(labeled_data_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                all_blocks.extend(json.load(f))
    
    if not all_blocks:
        print(f"Error: No labeled JSON files found in '{labeled_data_path}'.")
        return

    # ## CRITICAL CHANGE ##: Use the lightweight FeatureExtractorLite class
    print("Initializing LITE feature extractor (no sentence model)...")
    feature_extractor = FeatureExtractorLite()
    
    # This will be very fast because it does not generate embeddings.
    feature_matrix, font_map = feature_extractor.extract_features(all_blocks)
    labels = np.array([LABEL_MAPPING.get(block.get('label', 'NONE'), 0) for block in all_blocks])

    print(f"Training on {feature_matrix.shape[0]} samples with {feature_matrix.shape[1]} features.")

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(LABEL_MAPPING),
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(feature_matrix, labels)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(model_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Bundle the trained model with its metadata
    model_bundle = {
        'model': model,
        'font_map': font_map,
        'label_mapping': LABEL_MAPPING,
        'feature_names': feature_extractor.feature_names
    }
    joblib.dump(model_bundle, model_output_path)
    print(f"Training complete. Model bundle saved to: {model_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Outline Extraction Pipeline")
    parser.add_argument('phase', choices=['extract', 'train'], help="The pipeline phase to run.")
    parser.add_argument('--input', required=True, help="Input file (for extract) or directory (for train).")
    parser.add_argument('--output', required=True, help="Output file path for the generated data or model.")
    
    args = parser.parse_args()

    if args.phase == 'extract':
        run_extraction(args.input, args.output)
    elif args.phase == 'train':
        run_training(args.input, args.output)