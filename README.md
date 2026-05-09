# DDR Report Generator

A Python-based system that automatically generates **Detailed Diagnostic Reports (DDR)** from building inspection and thermal imaging data.

## What it does
Takes two input documents:
- **Inspection Report** (PDF) — visual site observations, area-wise issue descriptions
- **Thermal Report** (PDF) — thermal imaging data with temperature readings

And produces a structured DDR report (HTML + Markdown) that merges both data sources into a client-ready diagnostic document.

## How it works

1. **PDF Extraction** — Uses PyMuPDF to extract text and images from both input PDFs
2. **Image Processing** — Filters and catalogs relevant images with page-level context
3. **AI Analysis** — Sends extracted data to Gemini for structured diagnostic reasoning
4. **Report Generation** — Produces a professional HTML report with images embedded in the correct sections

## DDR Output Structure

The generated report follows this structure:
1. Property Issue Summary
2. Area-wise Observations (with embedded images)
3. Probable Root Cause
4. Severity Assessment (Critical / High / Medium / Low)
5. Recommended Actions
6. Additional Notes
7. Missing or Unclear Information

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

Place your input PDFs in the project root:
- `Sample Report.pdf` (inspection data)
- `Thermal Images.pdf` (thermal data)

Run:
```bash
python ddr_generator.py
```

Output will be saved to `output/DDR_Report.html` and `output/DDR_Report.md`.

## Design Decisions

- **Image filtering**: Small images (< 200x200px) are skipped since they're usually decorative elements or icons, not diagnostic photos
- **Missing data handling**: If information is absent from source documents, the report explicitly marks it as "Not Available" rather than guessing
- **Conflict detection**: When inspection and thermal data contradict each other, the system flags the conflict instead of silently picking one
- **Model-agnostic**: The LLM layer can be swapped to any provider (GPT-4, Claude, etc.) by changing the model configuration

## Tech Stack

- Python 3.x
- PyMuPDF (fitz) — PDF text and image extraction
- Google Gemini AI — structured reasoning and report generation
- Pillow — image validation and processing
