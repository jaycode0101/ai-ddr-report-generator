# DDR Report Generator — Assignment Submission
## Jayesh Shewale

---

### Links

- **GitHub Repository:** https://github.com/jaycode0101/ai-ddr-report-generator
- **Loom Video:** [PASTE YOUR LOOM LINK HERE AFTER RECORDING]

---

### What I Built

A Python-based AI system that reads two input PDFs (Inspection Report + Thermal Report), extracts text and images using PyMuPDF, processes them through Google Gemini AI, and generates a structured Detailed Diagnostic Report (DDR) with images embedded in the correct sections.

### Tech Stack

- Python 3.x
- PyMuPDF (fitz) — PDF text and image extraction
- Google Gemini AI — structured reasoning and report generation
- Pillow — image validation

### How to Run

```bash
pip install -r requirements.txt
# Add your Gemini API key to .env file
python ddr_generator.py
```

Output is generated at `output/DDR_Report.html` and `output/DDR_Report.pdf`.

### DDR Output Sections

1. Property Issue Summary
2. Area-wise Observations (with embedded images)
3. Probable Root Cause
4. Severity Assessment (Critical/High/Medium/Low)
5. Recommended Actions
6. Additional Notes
7. Missing or Unclear Information

### Key Design Decisions

- Images smaller than 200px are filtered out to avoid extracting decorative PDF elements
- Missing information is explicitly flagged as "Not Available"
- Conflicting data between reports is flagged (e.g., structural score vs crack observations)
- System is generalizable — works on any similar inspection + thermal report pair
