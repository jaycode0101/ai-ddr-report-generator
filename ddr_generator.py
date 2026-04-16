# ddr_generator.py
# Reads inspection + thermal PDFs and generates a merged DDR report

import os
import sys
import re
import time
import base64
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found in environment or .env file.")
    print("Create a .env file with: GEMINI_API_KEY=your_key_here")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Constants
MAX_IMAGES = 120         # Cap extracted images to avoid massive prompts
MIN_IMAGE_SIZE = 200     # Minimum width/height in pixels
MAX_TEXT_CHARS = 6000     # Max chars per document to send to AI
MAX_RETRIES = 3
RETRY_DELAY = 30         # seconds


class PDFExtractor:
    # handles pulling text and images out of a PDF

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.filename = Path(pdf_path).stem

    def extract_text(self):
        # grab all text content page by page
        full_text = ""
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            if text.strip():
                full_text += f"\n--- Page {page_num + 1} ---\n"
                full_text += text
        return full_text

    def extract_images(self, output_dir):
        # pull out images and save them with some context info
        os.makedirs(output_dir, exist_ok=True)
        images_metadata = []
        img_count = 0

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)

            # Also get the text on this page for context
            page_text = page.get_text().strip()
            # Take first 200 chars as context
            page_context = page_text[:200] if page_text else "No text context"

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = self.doc.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        img_count += 1

                        # Save image
                        img_filename = f"{self.filename}_page{page_num + 1}_img{img_count}.{image_ext}"
                        img_path = os.path.join(output_dir, img_filename)

                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        # Convert small images to check if they're useful (skip tiny icons)
                        try:
                            pil_img = Image.open(io.BytesIO(image_bytes))
                            width, height = pil_img.size
                            if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                                os.remove(img_path)
                                continue
                        except Exception:
                            pass

                        images_metadata.append({
                            "filename": img_filename,
                            "path": img_path,
                            "page": page_num + 1,
                            "source": self.filename,
                            "page_context": page_context,
                            "dimensions": f"{base_image.get('width', 'N/A')}x{base_image.get('height', 'N/A')}"
                        })
                except Exception as e:
                    print(f"  [Warning] Could not extract image {img_index} from page {page_num + 1}: {e}")

        # Cap at MAX_IMAGES to avoid massive prompts
        if len(images_metadata) > MAX_IMAGES:
            print(f"  [Info] Capping images from {len(images_metadata)} to {MAX_IMAGES}")
            images_metadata = images_metadata[:MAX_IMAGES]
        return images_metadata

    def close(self):
        self.doc.close()


class DDRGenerator:
    # main class - takes inspection + thermal PDFs and produces the DDR

    def __init__(self, inspection_pdf, thermal_pdf, reference_ddr=None):
        self.inspection_pdf = inspection_pdf
        self.thermal_pdf = thermal_pdf
        self.reference_ddr = reference_ddr
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.output_dir = os.path.join(os.path.dirname(inspection_pdf), "output")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

    def extract_all_data(self):
        # run extraction on all input PDFs
        print("\n[1/4] Extracting data from Inspection Report...")
        insp_extractor = PDFExtractor(self.inspection_pdf)
        self.inspection_text = insp_extractor.extract_text()
        self.inspection_images = insp_extractor.extract_images(self.images_dir)
        insp_extractor.close()
        print(f"  -> Extracted {len(self.inspection_text)} chars of text")
        print(f"  -> Extracted {len(self.inspection_images)} images")

        print("\n[2/4] Extracting data from Thermal Report...")
        thermal_extractor = PDFExtractor(self.thermal_pdf)
        self.thermal_text = thermal_extractor.extract_text()
        self.thermal_images = thermal_extractor.extract_images(self.images_dir)
        thermal_extractor.close()
        print(f"  -> Extracted {len(self.thermal_text)} chars of text")
        print(f"  -> Extracted {len(self.thermal_images)} images")

        # Reference DDR (optional - for format guidance)
        self.reference_text = ""
        if self.reference_ddr and os.path.exists(self.reference_ddr):
            print("\n[2.5/4] Reading reference DDR for format guidance...")
            ref_extractor = PDFExtractor(self.reference_ddr)
            self.reference_text = ref_extractor.extract_text()
            ref_extractor.close()
            # Trim reference to save tokens
            self.reference_text = self.reference_text[:3000]
            print(f"  -> Reference DDR loaded (trimmed to {len(self.reference_text)} chars)")

        # Combine all images
        self.all_images = self.inspection_images + self.thermal_images
        print(f"\n  Total images extracted: {len(self.all_images)}")

    def _build_image_catalog(self):
        # build a text list of images so the LLM knows what's available
        catalog = "## Extracted Images Catalog\n\n"
        for i, img in enumerate(self.all_images):
            catalog += f"Image #{i+1}:\n"
            catalog += f"  - Filename: {img['filename']}\n"
            catalog += f"  - Source: {img['source']}\n"
            catalog += f"  - Page: {img['page']}\n"
            catalog += f"  - Dimensions: {img['dimensions']}\n"
            catalog += f"  - Page Context: {img['page_context'][:150]}\n\n"
        return catalog

    def generate_ddr_content(self):
        # send everything to gemini and get the DDR back
        print("\n[3/4] Generating DDR with Gemini AI...")

        image_catalog = self._build_image_catalog()

        # Trim input texts to fit within token limits
        insp_text = self.inspection_text[:MAX_TEXT_CHARS]
        therm_text = self.thermal_text[:MAX_TEXT_CHARS]

        prompt = f"""You are a Senior Building Diagnostics Engineer. Generate a DDR (Detailed Diagnostic Report) by merging the INSPECTION REPORT and THERMAL REPORT below.

RULES:
- Do NOT invent facts not in the documents
- If info conflicts between reports, mention the conflict
- If info is missing, write "Not Available"
- Use simple client-friendly language
- Reference images as [IMAGE: #X - description]
- For thermal images: SUMMARIZE temperature ranges per area (e.g. "Temperature range: 20.2°C to 28.8°C") instead of listing every single reading individually. Only reference 2-3 representative thermal images per area, not all of them.
- Only reference image numbers that exist in the IMAGE CATALOG below. Do NOT reference image numbers higher than the maximum in the catalog.

DDR STRUCTURE (use these exact section headings):
1. PROPERTY ISSUE SUMMARY
2. AREA-WISE OBSERVATIONS (include image references per area)
3. PROBABLE ROOT CAUSE
4. SEVERITY ASSESSMENT (Critical/High/Medium/Low with reasoning)
5. RECOMMENDED ACTIONS
6. ADDITIONAL NOTES
7. MISSING OR UNCLEAR INFORMATION

INSPECTION REPORT:
{insp_text}

THERMAL REPORT:
{therm_text}

IMAGE CATALOG:
{image_catalog}

Generate the complete DDR now in markdown format."""

        print(f"  -> Prompt size: {len(prompt)} chars")

        # Retry logic for quota limits
        for attempt in range(MAX_RETRIES):
            try:
                response = self.model.generate_content(prompt)
                self.ddr_content = response.text
                print(f"  -> DDR content generated ({len(self.ddr_content)} chars)")
                return self.ddr_content
            except Exception as e:
                error_str = str(e)
                if '429' in error_str or 'ResourceExhausted' in error_str:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  [Rate Limited] Waiting {wait}s before retry {attempt+2}/{MAX_RETRIES}...")
                    time.sleep(wait)
                else:
                    raise e

        print("[ERROR] Failed after all retries. Quota may be fully exhausted.")
        print("  -> Try again in a few minutes or add billing to your Google AI project.")
        sys.exit(1)

    def build_html_report(self):
        # convert the markdown DDR into a styled HTML page with images
        print("\n[4/4] Building final HTML report with embedded images...")

        # Parse image references from DDR content and prepare embedded images
        image_html_map = {}
        for i, img in enumerate(self.all_images):
            img_num = i + 1
            try:
                with open(img['path'], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                ext = img['filename'].split('.')[-1]
                mime = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
                image_html_map[f"#{ img_num}"] = f"""
                <div class="report-image">
                    <img src="data:{mime};base64,{img_data}" alt="Image from {img['source']} Page {img['page']}">
                    <p class="image-caption">Source: {img['source']} | Page {img['page']}</p>
                </div>
                """
            except Exception as e:
                image_html_map[f"#{img_num}"] = f'<p class="image-missing">Image #{img_num}: Could not load — {str(e)}</p>'

        # Convert markdown-style content to HTML
        ddr_html = self.ddr_content

        # Replace image references [IMAGE: #X - description] with actual images
        import re
        def replace_image_ref(match):
            full_match = match.group(0)
            # Extract the image number
            num_match = re.search(r'#(\d+)', full_match)
            if num_match:
                img_key = f"#{num_match.group(1)}"
                if img_key in image_html_map:
                    return image_html_map[img_key]
            return f'<p class="image-missing">{full_match} — Image Not Available</p>'

        ddr_html = re.sub(r'\[IMAGE:.*?\]', replace_image_ref, ddr_html)

        # Convert markdown headers to HTML
        ddr_html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', ddr_html, flags=re.MULTILINE)
        ddr_html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', ddr_html, flags=re.MULTILINE)
        ddr_html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', ddr_html, flags=re.MULTILINE)

        # Convert bold
        ddr_html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ddr_html)

        # Convert bullet points
        ddr_html = re.sub(r'^\- (.+)$', r'<li>\1</li>', ddr_html, flags=re.MULTILINE)
        ddr_html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', ddr_html)

        # Convert newlines to paragraphs
        paragraphs = ddr_html.split('\n\n')
        ddr_html = '\n'.join([f'<p>{p}</p>' if not p.strip().startswith('<') else p for p in paragraphs])

        # Build final HTML
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DDR - Detailed Diagnostic Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: #f8f9fa;
            color: #2d3748;
            line-height: 1.7;
            padding: 20px;
        }}

        .report-container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 60px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-radius: 8px;
        }}

        .report-header {{
            text-align: center;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}

        .report-header h1 {{
            font-size: 28px;
            font-weight: 700;
            color: #1e3a5f;
            margin-bottom: 8px;
        }}

        .report-header .subtitle {{
            font-size: 14px;
            color: #64748b;
        }}

        .report-header .generated-date {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 10px;
        }}

        h1 {{
            font-size: 24px;
            color: #1e3a5f;
            margin: 35px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }}

        h2 {{
            font-size: 20px;
            color: #2563eb;
            margin: 30px 0 12px 0;
        }}

        h3 {{
            font-size: 16px;
            color: #475569;
            margin: 20px 0 10px 0;
        }}

        p {{
            margin: 8px 0;
            font-size: 14px;
        }}

        ul {{
            margin: 10px 0 10px 25px;
        }}

        li {{
            margin: 5px 0;
            font-size: 14px;
        }}

        strong {{
            color: #1e3a5f;
        }}

        .report-image {{
            margin: 20px 0;
            text-align: center;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            overflow: hidden;
            background: #f8fafc;
        }}

        .report-image img {{
            max-width: 100%;
            height: auto;
            max-height: 400px;
            object-fit: contain;
        }}

        .image-caption {{
            font-size: 11px;
            color: #94a3b8;
            padding: 8px;
            background: #f1f5f9;
            border-top: 1px solid #e2e8f0;
        }}

        .image-missing {{
            background: #fef2f2;
            color: #dc2626;
            padding: 12px;
            border-radius: 4px;
            font-size: 13px;
            border-left: 3px solid #dc2626;
            margin: 10px 0;
        }}

        .severity-critical {{
            background: #fef2f2;
            border-left: 4px solid #dc2626;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 0 4px 4px 0;
        }}

        .severity-high {{
            background: #fff7ed;
            border-left: 4px solid #ea580c;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 0 4px 4px 0;
        }}

        .severity-medium {{
            background: #fffbeb;
            border-left: 4px solid #d97706;
            padding: 10px 15px;
            margin: 8px 0;
            border-radius: 0 4px 4px 0;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #e2e8f0;
            text-align: center;
            font-size: 11px;
            color: #94a3b8;
        }}

        @media print {{
            body {{ padding: 0; background: white; }}
            .report-container {{ box-shadow: none; padding: 30px; }}
            .report-image img {{ max-height: 300px; }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>DETAILED DIAGNOSTIC REPORT (DDR)</h1>
            <p class="subtitle">AI-Generated Property Health Assessment</p>
            <p class="generated-date">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p class="generated-date">System: Automated DDR Generator v1.0</p>
        </div>

        {ddr_html}

        <div class="footer">
            <p>This report was generated using an AI-powered diagnostic system.</p>
            <p>All findings are based strictly on the provided Inspection Report and Thermal Report.</p>
            <p>No information has been fabricated. Missing data is explicitly marked as "Not Available".</p>
        </div>
    </div>
</body>
</html>"""

        # Save the HTML report
        output_path = os.path.join(self.output_dir, "DDR_Report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n{'='*60}")
        print(f"  DDR REPORT GENERATED SUCCESSFULLY!")
        print(f"  Output: {output_path}")
        print(f"  Images: {self.images_dir}")
        print(f"  Total Images Embedded: {len(self.all_images)}")
        print(f"{'='*60}")

        return output_path

    def run(self):
        """Execute the full DDR generation pipeline."""
        print("=" * 60)
        print("  DDR REPORT GENERATOR — INITIALIZING")
        print("=" * 60)

        self.extract_all_data()
        self.generate_ddr_content()

        # Save raw markdown version too
        md_path = os.path.join(self.output_dir, "DDR_Report.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.ddr_content)
        print(f"  -> Markdown version saved: {md_path}")

        output_path = self.build_html_report()
        return output_path


if __name__ == "__main__":
    # Default paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    INSPECTION_PDF = os.path.join(BASE_DIR, "Sample Report.pdf")
    THERMAL_PDF = os.path.join(BASE_DIR, "Thermal Images.pdf")
    REFERENCE_DDR = os.path.join(BASE_DIR, "Main DDR.pdf")

    # Validate files exist
    for path, name in [(INSPECTION_PDF, "Inspection Report"), (THERMAL_PDF, "Thermal Report")]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} not found at: {path}")
            sys.exit(1)

    # Initialize and run
    generator = DDRGenerator(
        inspection_pdf=INSPECTION_PDF,
        thermal_pdf=THERMAL_PDF,
        reference_ddr=REFERENCE_DDR if os.path.exists(REFERENCE_DDR) else None
    )

    output = generator.run()
    print(f"\n  Open this file in your browser to view the DDR:")
    print(f"  {output}")
