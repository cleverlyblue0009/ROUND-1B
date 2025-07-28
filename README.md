# Adobe2 Persona-Driven Document Intelligence

This module analyzes PDFs based on a persona and job-to-be-done, extracting and ranking relevant sections and snippets.

## Features

- Accepts a task JSON describing persona, job-to-be-done, and input documents.
- Extracts logical sections and headings from PDFs using outline_core.py.
- Ranks sections by relevance to the persona and job using scoring.py (TF-IDF, keyword overlap, etc.).
- Extracts and ranks relevant text snippets (subsections) from top sections.
- Outputs a single output.json with metadata, ranked sections, and snippet analysis.

## How to Run

### 1. (Optional) Create and activate a virtual environment

bash
python3 -m venv venv
source venv/bin/activate


### 2. Install dependencies

bash
pip install -r requirements.txt


### 3. Place your input files

- Put your PDF files and a task.json (see below) in the input/ directory inside adobe2/.

### 4. Run the script

bash
python main_b.py --input input --output output --task task.json


### 5. Output

- The script will generate output.json in the output/ directory.

## Input/Output Example

### Example task.json (input format)

json
{
  "challenge_info": {},
  "documents": [
    {"filename": "file01.pdf", "title": "Sample Doc"}
  ],
  "persona": { "role": "Travel Planner" },
  "job_to_be_done": { "task": "Plan a 4-day trip..." }
}


### Example output.json (output format)

json
{
  "metadata": {
    "input_documents": ["file01.pdf"],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a 4-day trip...",
    "processing_timestamp": "2024-06-07T12:34:56"
  },
  "extracted_sections": [
    {
      "document": "file01.pdf",
      "section_title": "Introduction",
      "importance_rank": 1,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "file01.pdf",
      "refined_text": "Relevant snippet...",
      "page_number": 2
    }
  ]
}


## Dependencies

- pymupdf
- numpy
- scikit-learn

Install with:
bash
pip install -r requirements.txt
