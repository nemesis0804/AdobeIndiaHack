PDF Outline & Heading Extractor
This project, developed for the Adobe India Hackathon, is a powerful tool designed to intelligently parse PDF documents and extract a structured, hierarchical outline. It addresses the common challenge of unstructured PDFs by using a hybrid intelligence approach, combining a lightweight machine learning model with an advanced, context-aware rule engine to accurately identify titles and heading levels (H1, H2, H3, etc.).
PROJECT STRUCTURE
For clean and scalable development, this project uses a separated structure for code and data.
ADOBE1A/ (The Code Repository): This folder contains all the Python scripts (predict.py, main.py), utilities, and the trained model. You run all commands from here.
adobeindiahackathon/ (The Data Folder): This external folder holds your data.
pdfsinput/: Drop all your source PDF files here.
output/: The script will automatically save the corresponding JSON results here.
SETUP AND INSTALLATION
Clone the Repository:
Generated bash->git clone https://github.com/nemesis0804/AIH.git
cd AIH
Bash
Install Dependencies: Ensure you have Python 3.8+ installed. Then, install all required libraries using the requirements.txt file.
Generated bash->pip install -r requirements.txt
HOW TO RUN PREDICTIONS
This tool is designed for batch processing. You can place multiple PDFs in the input folder and process them all with a single command.
Place PDFs: Add one or more PDF files into your data folder, e.g., C:\Users\YourUser\Desktop\adobeindiahackathon\pdfsinput.
Execute the Script: Open a terminal inside the code repository (AIH/ or ADOBE1A/) and run the process_all_pdfs.py script. You must provide the full paths to your model, input directory, and output directory.
GENERATED BASH->python process_all_pdfs.py --model "models\heading_model.joblib" --input_dir "C:\Users\YourUser\Desktop\adobeindiahackathon\pdfsinput" --output_dir "C:\Users\YourUser\Desktop\adobeindiahackathon\output"
This script will iterate through every PDF in the pdfsinput folder and save a corresponding .json file in the output folder.
