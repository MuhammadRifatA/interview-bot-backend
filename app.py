from flask import Flask, jsonify, request
from huggingface_hub import InferenceClient
from flask_cors import CORS
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import faiss
from flask_sqlalchemy import SQLAlchemy
import json
import os
from sqlalchemy import create_engine
import re

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200", "https://muhammadrifata.github.io"], supports_credentials=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://interview_bot_postgres_user:509OIoHEardtpBjC51OQARuzaF6CFGqY@dpg-d12m4895pdvs73csp5h0-a.singapore-postgres.render.com/interview_bot_postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

MODEL_PATHS = {
    "similarity": "models/all-MiniLM-L6-v2",
}

similarity_model = SentenceTransformer(MODEL_PATHS["similarity"])

class Interview(db.Model):
    __tablename__ = 'interview'
    qid = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    generated_answer = db.Column(db.Text)
    feedback = db.Column(db.Text)
    score = db.Column(db.Integer)
    session_id = db.Column(db.String(64))

def get_category_from_db(qid):
    question_obj = Interview.query.filter_by(qid=qid).first()
    if question_obj:
        return question_obj.category
    return "Other"

def calculate_cosine_similarity(answer_1, answer_2):
    answer_1_embedding = similarity_model.encode([answer_1], convert_to_numpy=True, normalize_embeddings=True)
    answer_2_embedding = similarity_model.encode([answer_2], convert_to_numpy=True, normalize_embeddings=True)
    cosine_similarity = util.cos_sim(answer_1_embedding, answer_2_embedding)[0][0].item()
    return cosine_similarity

def retrieve_information(query, documents, index):
    query_embedding = similarity_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(query_embedding, k=1)
    retrieved_doc = documents[I[0][0]]
    return retrieved_doc

def generate_feedback_and_score(question, user_answer, documents, index, category):
    client = InferenceClient(
        provider="novita",
        api_key = os.environ.get("HF_API_KEY"),
    )

    relevant_doc = retrieve_information(question, documents, index)

    prompt = f"""
    You are an interview assistant. Evaluate the candidate's answer to this **specific** interview question **only**.

    - Score from 1 to 10 based on how well the answer addresses the question.
    - Focus on clarity, relevance, and completeness.
    - Do not assume or infer a different question.
    - Base your comparison using the context provided.

    Here is the input:

    Question: "{question}"

    Candidate answer: "{user_answer}"

    Reference context: "{relevant_doc}"
    """

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        n=3
    )

    feedback_list = []
    for choice in completion.choices:
        content = choice.message['content'].strip()
        score = 1
        try:
            response_json = json.loads(content)
            score = int(response_json.get("score", 1))
            feedback = response_json.get("feedback", content)
        except json.JSONDecodeError:
            match = re.search(r"score\s*[:\-]?\s*(\d+)", content, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                feedback = content
            else:
                feedback = content

        feedback = feedback.replace('\n', ' ').replace('*', '').strip()
        feedback_list.append((score, feedback))

    # Ambil feedback terbaik
    llm_score, llm_feedback = max(feedback_list, key=lambda x: x[0])

    # Simpan semua feedback ke log
    with open("log_pengujian_rag.txt", "a", encoding="utf-8") as f:
        f.write("\n=== Pengujian Baru ===\n")
        f.write(f"Pertanyaan: {question}\n")
        f.write(f"Retrieved Document: {relevant_doc}\n")
        f.write(f"Jawaban User: {user_answer}\n")
        for i, (score, feedback) in enumerate(feedback_list, start=1):
            f.write(f"Pilihan LLM #{i} - Skor: {score}, Feedback: {feedback}\n")
        f.write(f"✅ Feedback Terbaik Dipilih: {llm_feedback} (Skor: {llm_score})\n")

    return llm_score, llm_feedback

# weight_sim=bobot consine, weight_llm=bobot llm, sim_score=similiarty score, consine_sim=consine similiarty 
def score_with_rubric(cosine_sim, llm_score):
    sim_score = cosine_sim * 10
    weight_sim = 0.5
    weight_llm = 0.5
    final_score = weight_sim * sim_score + weight_llm * llm_score
    return round(final_score, 2)

@app.route('/')
def home():
    questions = Interview.query.with_entities(Interview.qid, Interview.question).all()
    random.shuffle(questions)

    translated_questions = [
        {
            "qid": q.qid,
            "question": q.question
        }
        for q in questions[:2]
    ]

    return jsonify(translated_questions)

@app.route('/jawab/<int:idx>', methods=['POST', 'OPTIONS'])
def run_question(idx):
    if idx < 1:
        return jsonify({"error": "ID pertanyaan harus dimulai dari 1."}), 400

    if request.method == 'OPTIONS':
        return '', 200

    question_obj = Interview.query.filter_by(qid=idx).first()
    if not question_obj:
        return jsonify({"error": f"Pertanyaan dengan ID {idx} tidak tersedia."}), 404

    category = get_category_from_db(idx)
    question = question_obj.question
    user_answer = request.json.get("answer", "").strip()

    if not user_answer:
        return jsonify({"error": f"Pertanyaan {idx} selesai. Anda tidak memberikan jawaban."}), 400

    # Validasi minimal 5 kata
    if len(user_answer.split()) < 5:
        return jsonify({"error": "Jawaban Anda terlalu singkat."}), 400

    user_answer = request.json.get("answer", "").strip()
    session_id = request.json.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    # Siapkan dokumen untuk pencarian relevansi
    documents = [q.question for q in Interview.query.all()]
    document_embeddings = similarity_model.encode(documents, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(document_embeddings)

    # Panggil LLM untuk mendapatkan score dan feedback
    llm_score, llm_feedback = generate_feedback_and_score(question, user_answer, documents, index, category)

    # Bersihkan feedback:
    # - Hapus "score: x" jika muncul di awal
    # - Hapus karakter '\n' dan '*'
    llm_feedback = re.sub(r'^["\']?score["\']?\s*:\s*\d+\s*[,;:-]*', '', llm_feedback, flags=re.IGNORECASE)
    llm_feedback = llm_feedback.replace('\n', ' ').replace('*', '').strip()

    # Hitung cosine similarity dengan jawaban referensi
    reference_answer = question_obj.answer or "Jawaban referensi yang ideal sesuai pertanyaan ini."
    cosine_sim = calculate_cosine_similarity(user_answer, reference_answer)

    # Skor akhir gabungan
    final_score = score_with_rubric(cosine_sim, llm_score)

    # Gabungkan feedback akhir
    combined_feedback = f"{llm_feedback} Skor Akhir Anda: {final_score} / 10"

    # Simpan ke database
    question_obj.generated_answer = user_answer
    question_obj.feedback = json.dumps(combined_feedback)
    question_obj.score = final_score
    question_obj.session_id = session_id 
    db.session.commit()

    return jsonify({
        "message": f"Pertanyaan {idx} selesai.",
        "feedback": combined_feedback,
    })

@app.route('/hasil', methods=['GET', 'OPTIONS'])
def show_result():
    if request.method == 'OPTIONS':
        return '', 200

    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Session ID is required."}), 400

    data = Interview.query.filter(
        Interview.session_id == session_id,
        Interview.generated_answer.isnot(None)  ,
        Interview.score.isnot(None)
    ).all()

    if not data:
        return jsonify({"message": "Anda belum menjawab pertanyaan apapun untuk sesi ini."})

    total_score = sum(row.score for row in data)
    max_score = len(data) * 10
    passing_score = max_score * 0.6

    result = {
        "jumlah_pertanyaan_dijawab": len(data),
        "total_score": total_score,
        "max_score": max_score,
        "status": "✅ Passed" if total_score >= passing_score else "❌ Not Passed",
        "detail": [
            {
                "qid": row.qid,
                "question": row.question,
                "answer": row.generated_answer,
                "score": row.score,
                "feedback": json.loads(row.feedback) if row.feedback else ""
            } for row in data
        ]
    }

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
