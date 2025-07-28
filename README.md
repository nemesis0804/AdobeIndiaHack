PDF Outline & Heading Extractor (Dockerized)
This project, developed for the Adobe India Hackathon, is a powerful, containerized application designed to intelligently parse PDF documents and extract a structured, hierarchical outline. It solves the common challenge of unstructured PDFs by using a hybrid intelligence approach, combining a lightweight XGBoost machine learning model with an advanced, context-aware rule engine to accurately identify titles and heading levels (H1, H2, H3, etc.).
The entire application is wrapped in a Docker container, ensuring a consistent, dependency-free, and "it just works" deployment on any machine with Docker installed.
HOW TO START
Follow these steps to set up the project and run the application.
Prerequisites
Git
Docker Desktop installed and running.
1. Clone the Repository
First, clone the project code to your local machine.
Generated bash->git clone https://github.com/nemesis0804/AIH.git
cd AIH
2. Create Local Data Directories
Inside the cloned project folder (AIH/), create two new directories: input and output.
input/: You will place your source PDF files here.
output/: The JSON results will be automatically saved here.
3. Build the Docker Image
Navigate into the project root in your terminal (AIH/) and run the docker build command. This will create a self-contained image named adobehack1a with all Python dependencies included.
Generated bash->docker build -t adobehack1a .
How to Run the Application
The application is run via the docker run command, which starts the container and links it to your local input and output folders.
Place PDFs: Add one or more PDF files into the input/ folder you just created.
Run the Container: Open a terminal in the project's root directory (AIH/). The command is slightly different depending on your terminal.
For Windows PowerShell:
Generated powershell
docker run --rm `
  -v "$(Resolve-Path ./input):/app/input" `
  -v "$(Resolve-Path ./output):/app/output" `
  adobehack1a `
  python process_all_pdfs.py `
  --input_dir /app/input `
  --output_dir /app/output `
  --model models/heading_model.joblib
   For Command Prompt (cmd) or other terminals (macOS/Linux):
Generated bash
docker run --rm ^
  -v "%cd%\input:/app/input" ^
  -v "%cd%\output:/app/output" ^
  adobehack1a ^
  python process_all_pdfs.py ^
  --input_dir /app/input ^
  --output_dir /app/output ^
  --model models/heading_model.joblib
   
   The container will start, process every PDF in your input folder, save the corresponding JSON files to your output folder, and then automatically stop and clean up after itself.
