import streamlit as st
import random
import numpy as np
import json
import os
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Interview Simulator", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body, .main {
    background-color: #0f172a;
}
h1, h2, h3 {
    color: #38bdf8;
}
textarea {
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
.metric-box {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠🎤 AI Interview Simulator (Offline)")

# ------------------- LOAD VOSK MODEL -------------------
@st.cache_resource
def load_vosk():
    model_path = os.path.join(os.getcwd(), "vosk-model-small-en-us-0.15")
    if not os.path.exists(model_path):
        st.error("❌ Vosk model folder not found")
        st.stop()
    return Model(model_path)

vosk_model = load_vosk()

# ------------------- VOICE TO TEXT -------------------
def record_and_transcribe(duration=8, samplerate=16000):
    recognizer = KaldiRecognizer(vosk_model, samplerate)

    st.info("🎙️ Speak now...")

    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='int16'
    )
    sd.wait()

    # ✅ FIX: convert numpy array to raw bytes
    recognizer.AcceptWaveform(audio.tobytes())

    result = json.loads(recognizer.Result())
    return result.get("text", "")

# ------------------- UPDATED SKILL BOARD -------------------
QUESTIONS = {
    "Python": [
        "What is Python?",
        "What are variables?",
        "Difference between list and tuple?",
        "What is a function?",
        "What is a loop?"
    ],
    "Machine Learning": [
        "What is Machine Learning?",
        "What is supervised learning?",
        "What is overfitting?",
        "What is a dataset?",
        "What is accuracy?"
    ],
    "Data Science": [
        "What is Data Science?",
        "What is data cleaning?",
        "What is data visualization?",
        "What is Pandas?",
        "What is NumPy?"
    ],
    "HTML": [
        "What is HTML?",
        "What is a tag?",
        "Difference between div and span?",
        "What is a form?",
        "What is semantic HTML?"
    ],
    "CSS": [
        "What is CSS?",
        "Difference between class and id?",
        "What is box model?",
        "What is flexbox?",
        "What is responsive design?"
    ],
    "JavaScript": [
        "What is JavaScript?",
        "Difference between var, let and const?",
        "What is DOM?",
        "What is an event?",
        "Difference between == and ===?"
    ]
}

# ------------------- QUESTION GENERATOR -------------------
def generate_question(skill, difficulty):
    q = random.choice(QUESTIONS[skill])
    if difficulty == "Advanced":
        return f"Explain in detail: {q}"
    elif difficulty == "Intermediate":
        return f"Explain with example: {q}"
    return q

# ------------------- SCORING ENGINE -------------------
def score_answer(answer, question):
    if len(answer.strip()) < 5:
        return 0.1
    corpus = [question, answer]
    tfidf = TfidfVectorizer()
    vec = tfidf.fit_transform(corpus)
    similarity = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    length_score = min(len(answer.split()) / 40, 1)
    return round(similarity * 0.6 + length_score * 0.4, 2)

def confidence_score(answer):
    words = len(answer.split())
    filler = answer.lower().count("uh") + answer.lower().count("um")
    return round(max(min(words / 50 - filler * 0.1, 1), 0), 2)

# ------------------- SESSION STATE -------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "question" not in st.session_state:
    st.session_state.question = None

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("🎯 Interview Setup")
    role = st.selectbox("Role", ["Software Engineer", "Data Scientist", "Frontend Developer"])
    skill = st.selectbox("Skill", list(QUESTIONS.keys()))
    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

# ------------------- QUESTION -------------------
if st.session_state.question is None:
    st.session_state.question = generate_question(skill, difficulty)

st.markdown(f"""
<div style="background:#020617;padding:20px;border-radius:12px;border-left:6px solid #38bdf8;font-size:18px">
❓ <b>{st.session_state.question}</b>
</div>
""", unsafe_allow_html=True)

# ------------------- ANSWER INPUT -------------------
col1, col2 = st.columns([3, 1])

with col1:
    answer = st.text_area("Your Answer", height=160)

with col2:
    if st.button("🎙️ Speak"):
        answer = record_and_transcribe()
        st.success("Voice captured")
        st.text_area("Captured Answer", answer, height=160)

# ------------------- SUBMIT -------------------
if st.button("Submit Answer"):
    score = score_answer(answer, st.session_state.question)
    confidence = confidence_score(answer)
    hiring = round((score * 0.7 + confidence * 0.3) * 100, 1)

    st.session_state.history.append({
        "skill": skill,
        "score": score,
        "confidence": confidence,
        "hiring": hiring
    })

    st.session_state.question = generate_question(skill, difficulty)
    st.success("✅ Answer evaluated. Next question loaded.")

# ------------------- DASHBOARD -------------------
if st.session_state.history:
    st.subheader("📊 Skill Performance Dashboard")

    skills = {}
    for h in st.session_state.history:
        skills.setdefault(h["skill"], []).append(h["score"])

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots()
        ax.bar(skills.keys(), [np.mean(v) for v in skills.values()])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Average Score")
        st.pyplot(fig)

    with colB:
        avg_hiring = np.mean([h["hiring"] for h in st.session_state.history])
        st.markdown(f"""
        <div class="metric-box">
            <h2>📈 Hiring Probability</h2>
            <h1>{avg_hiring:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

# ------------------- FINAL REPORT -------------------
if st.button("📄 Generate Interview Report"):
    st.subheader("📄 Final Interview Report")
    for i, h in enumerate(st.session_state.history, 1):
        st.write(f"**Q{i} – {h['skill']}**")
        st.write(f"Score: {h['score']} | Confidence: {h['confidence']} | Hiring: {h['hiring']}%")
        st.divider()