import pdfplumber
import pytesseract
from PIL import Image
import subprocess
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import textwrap

# -------------------------
# 1. Load PDF (Text + OCR)
# -------------------------
def load_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                pil_image = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil_image)
                text += ocr_text + "\n"
    return text

def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith("page") and not line.isdigit():
            cleaned.append(line)
    return " ".join(cleaned)

# -------------------------
# 2. Generate Questions
# -------------------------
def generate_questions(section_name, syllabus_text, count, marks, difficulty="mixed"):
    prompt = f"""
You are an expert exam paper setter.

Syllabus:
{syllabus_text}

Task:
Create {count} questions of {marks} marks each for {section_name}.
Cover all units fairly and do not repeat topics.
Use exam style numbering: Q1, Q2, etc.
Each question must end with "({marks} marks)".
"""
    result = subprocess.run(
        ["ollama", "run", "gemma:2b"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

# -------------------------
# 3. Save to PDF
# -------------------------
def save_to_pdf(filename, content):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50
    left_margin = 50
    max_width = width - 100
    wrapper = textwrap.TextWrapper(width=100)

    for paragraph in content.split("\n\n"):
        lines = textwrap.wrap(paragraph, width=100)
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(left_margin, y, line)
            y -= 15
        y -= 10  # Extra space between paragraphs

    c.save()

# -------------------------
# 4. Main Program
# -------------------------
if __name__ == "__main__":
    syllabus_path = "data/Syllabus.pdf"
    difficulty = input("⚡ Difficulty (easy / medium / hard / mixed): ")

    print("⏳ Loading syllabus...")
    raw_text = load_pdf(syllabus_path)
    syllabus_text = clean_text(raw_text)

    print("🤖 Generating Section A (10 × 2 marks)...")
    secA = generate_questions("Section A", syllabus_text, 10, 2, difficulty)

    print("🤖 Generating Section B (5 × 13 marks)...")
    secB = generate_questions("Section B", syllabus_text, 5, 13, difficulty)

    print("🤖 Generating Section C (1 × 15 marks)...")
    secC = generate_questions("Section C", syllabus_text, 1, 15, difficulty)

    final_paper = f"""\
QUESTION PAPER
====================

SECTION A (2 marks × 10 = 20 marks)

{secA}

SECTION B (13 marks × 5 = 65 marks)

{secB}

SECTION C (15 marks × 1 = 15 marks)

{secC}
"""

    save_to_pdf("Question_Paper.pdf", final_paper)
    print("✅ Question Paper generated: Question_Paper.pdf")
