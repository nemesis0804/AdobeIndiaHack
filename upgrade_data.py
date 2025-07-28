# Filename: upgrade_data.py

import os
import json
import argparse
import shutil
from langdetect import detect, lang_detect_exception
from langdetect import DetectorFactory

# Set a seed for langdetect for deterministic, repeatable results
DetectorFactory.seed = 0

def upgrade_dataset(directory_path):
    """
    Upgrades all JSON files in a directory by adding a 'language' key to each text block.
    This script is designed to be run once to update an existing labeled dataset.
    It will create backups of the original files before modifying them.
    """
    print(f"--- Starting Dataset Upgrade for directory: {directory_path} ---")

    # Create a backup directory
    backup_dir = os.path.join(directory_path, "pre_upgrade_backups")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Backups will be saved in: {backup_dir}")

    # Find all JSON files in the target directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            print(f"\nProcessing file: {filename}")

            # 1. Create a backup before doing anything
            try:
                backup_path = os.path.join(backup_dir, filename)
                shutil.copy2(file_path, backup_path)
                print(f"  -> Backup created at: {backup_path}")
            except Exception as e:
                print(f"  -> WARNING: Could not create backup for {filename}. Skipping. Error: {e}")
                continue

            # 2. Read, modify, and write the file
            try:
                # Read all existing blocks from the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    blocks = json.load(f)

                # Loop through each block to add the language key
                for block in blocks:
                    # Skip if this block has already been upgraded
                    if 'language' in block:
                        continue

                    text_to_detect = block.get('text', '')
                    detected_lang = 'unknown'

                    if text_to_detect:
                        try:
                            detected_lang = detect(text_to_detect)
                        except lang_detect_exception.LangDetectException:
                            # This happens on very short or ambiguous text
                            detected_lang = 'unknown'
                    
                    # Add the new key to the block
                    block['language'] = detected_lang
                
                # Write the entire modified list of blocks back to the same file
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Use ensure_ascii=False to keep Japanese/French characters readable
                    json.dump(blocks, f, indent=2, ensure_ascii=False)
                
                print(f"  -> Successfully upgraded {filename} with language data.")

            except Exception as e:
                print(f"  -> FATAL ERROR processing {filename}. Restore from backup if needed. Error: {e}")

    print("\n--- Dataset Upgrade Complete! ---")
    print("You can now re-train your model using the upgraded data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade existing labeled JSON files with language detection data.")
    parser.add_argument('input_dir', help="The directory containing the labeled JSON files to upgrade (e.g., data/labeled_training/).")
    
    args = parser.parse_args()
    upgrade_dataset(args.input_dir)