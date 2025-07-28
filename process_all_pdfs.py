# Filename: process_all_pdfs.py

import os
import argparse
# You must have 'predict.py' in the same directory for this import to work
from predict import predict_outline

def process_directory(input_dir, output_dir, model_path):
    """
    Processes all PDF files in a given directory, and saves the
    structured JSON output to a corresponding output directory.
    """
    print(f"--- Starting Batch Processing ---")
    print(f"Input PDF Directory: {input_dir}")
    print(f"Output JSON Directory: {output_dir}")
    print(f"Model Path: {model_path}")

    # Ensure the output directory exists. If not, create it.
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all files in the input directory
    try:
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    except FileNotFoundError:
        print(f"FATAL ERROR: The input directory '{input_dir}' was not found.")
        return

    if not pdf_files:
        print("Warning: No PDF files found in the input directory.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) to process: {pdf_files}\n")

    # Loop through each PDF file and process it
    for pdf_filename in pdf_files:
        input_pdf_path = os.path.join(input_dir, pdf_filename)
        
        base_filename = os.path.splitext(pdf_filename)[0]
        output_json_filename = f"{base_filename}.json"
        output_json_path = os.path.join(output_dir, output_json_filename)

        print(f"--- Processing: {pdf_filename} ---")

        try:
            # Call the function from predict.py
            predict_outline(
                pdf_path=input_pdf_path,
                model_path=model_path,
                output_path=output_json_path
            )
        except Exception as e:
            print(f"❌ An error occurred while processing {pdf_filename}: {e}")
            continue

    print("\n--- Batch Processing Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process all PDFs in a directory for heading extraction.")

    # These arguments are flexible and can point to any folder on your system
    parser.add_argument('--input_dir', required=True, help="Full path to the directory containing the PDF files.")
    parser.add_argument('--output_dir', required=True, help="Full path to the directory where output JSON files will be saved.")
    parser.add_argument('--model', required=True, help="Path to the saved model bundle (.joblib).")

    args = parser.parse_args()

    # Run the main processing function
    process_directory(args.input_dir, args.output_dir, args.model)