from flask import Flask, request, jsonify
from main import load_pdf, clean_text, generate_questions, save_to_pdf  # Import your functions

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get('message')

    # Example flow: Ask for difficulty first, then generate paper
    if "generate" in user_msg.lower():
        difficulty = "easy"  # You could parse the user's message here or store state
        syllabus_path = "data/Syllabus.pdf"
        raw_text = load_pdf(syllabus_path)
        syllabus_text = clean_text(raw_text)

        secA = generate_questions("Section A", syllabus_text, 10, 2, difficulty)
        secB = generate_questions("Section B", syllabus_text, 5, 13, difficulty)
        secC = generate_questions("Section C", syllabus_text, 1, 15, difficulty)

        final_paper = f"""\
QUESTION PAPER

SECTION A (2 marks × 10 = 20 marks)

{secA}

SECTION B (13 marks × 5 = 65 marks)

{secB}

SECTION C (15 marks × 1 = 15 marks)

{secC}
"""
        save_to_pdf("Question_Paper.pdf", final_paper)
        reply = "✅ Question paper has been generated as 'Question_Paper.pdf'."
    else:
        reply = "Hello! Type 'generate exam paper' to create a new paper."

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(port=5000)
