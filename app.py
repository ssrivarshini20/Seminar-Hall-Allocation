
import json
import re
from pathlib import Path
from datetime import datetime

import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "knowledge" / "AI_Seminar_Hall_Allocation.pdf"
HALLS_PATH = BASE_DIR / "data" / "halls.json"

# Assignment weights:
# H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)
W_UNUSED = 2
W_MISSING = 20
W_CLASH = 100
W_DISTANCE = 1


def load_halls():
    with open(HALLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdf_text():
    if not PDF_PATH.exists() or PdfReader is None:
        return ""
    reader = PdfReader(str(PDF_PATH))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)


def chunk_text(text, max_chars=1100):
    # Keep chunks reasonably small for retrieval.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


@st.cache_resource
def build_knowledge_base():
    text = extract_pdf_text()
    chunks = chunk_text(text)
    if not chunks:
        return [], None, None
    if TfidfVectorizer is None:
        return chunks, None, None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks)
    return chunks, vectorizer, matrix


def retrieve(query, top_k=3):
    chunks, vectorizer, matrix = build_knowledge_base()
    if not chunks:
        return []
    if vectorizer is None:
        return chunks[:top_k]
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).ravel()
    indices = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in indices if scores[i] > 0]


def normalize_facility(s):
    s = s.lower().strip()
    aliases = {
        "microphone": "mic",
        "microphone system": "mic",
        "projector": "projector",
        "air conditioning": "ac",
        "air-conditioning": "ac",
        "air conditioner": "ac",
        "ac": "ac",
        "mic": "mic",
    }
    return aliases.get(s, s)


def parse_facilities(text):
    facilities = []
    lower = text.lower()
    if "projector" in lower:
        facilities.append("projector")
    if "microphone" in lower or re.search(r"\bmic\b", lower):
        facilities.append("mic")
    if "air conditioning" in lower or "air-conditioning" in lower or re.search(r"\bac\b", lower):
        facilities.append("ac")
    return list(dict.fromkeys(facilities))


def parse_participants(text):
    patterns = [
        r"(?:participants?|people|students?|attendees?)\s*[:=]?\s*(\d+)",
        r"\bfor\s+(\d+)\s*(?:participants?|people|students?|attendees?)?\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return int(m.group(1))
    return None


def parse_time_range(text):
    # Accepts forms such as 10:00 AM - 12:00 PM, 10 AM to 12 PM, 14:00-16:00.
    pat = r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?\s*(?:-|to|–)\s*(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?"
    m = re.search(pat, text, re.I)
    if not m:
        return None

    h1, mi1, ap1, h2, mi2, ap2 = m.groups()
    ap1, ap2 = (ap1 or "").upper(), (ap2 or "").upper()
    mi1, mi2 = int(mi1 or 0), int(mi2 or 0)
    h1, h2 = int(h1), int(h2)

    if ap1 and not ap2:
        ap2 = ap1

    def to_minutes(h, mi, ap):
        if ap == "AM":
            h = 0 if h == 12 else h
        elif ap == "PM":
            h = 12 if h == 12 else h + 12
        return h * 60 + mi

    return to_minutes(h1, mi1, ap1), to_minutes(h2, mi2, ap2)


def booking_clash(hall, requested_range):
    if not requested_range:
        return 0
    req_start, req_end = requested_range
    clashes = 0
    for booking in hall.get("bookings", []):
        start_h, start_m = map(int, booking["start"].split(":"))
        end_h, end_m = map(int, booking["end"].split(":"))
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        if max(start, req_start) < min(end, req_end):
            clashes += 1
    return clashes


def score_hall(hall, participants, required_facilities, requested_range=None):
    unused = hall["capacity"] - participants
    missing = len([f for f in required_facilities if f not in hall["facilities"]])
    clash = booking_clash(hall, requested_range)
    distance = hall["distance_m"]

    score = (
        W_UNUSED * unused
        + W_MISSING * missing
        + W_CLASH * clash
        + W_DISTANCE * distance
    )
    return {
        "hall": hall["name"],
        "unused": unused,
        "missing": missing,
        "clash": clash,
        "distance": distance,
        "score": score,
    }


def allocate(participants, required_facilities, requested_range=None):
    halls = load_halls()
    eligible = [h for h in halls if h["capacity"] >= participants]
    if not eligible:
        return [], "No hall has enough capacity for the requested number of participants."

    scored = [
        score_hall(h, participants, required_facilities, requested_range)
        for h in eligible
    ]
    scored.sort(key=lambda x: x["score"])
    return scored, None


def hill_climbing_demo(participants, required_facilities, requested_range=None):
    # A small demonstration of the assignment's Hill Climbing idea.
    # State = selected hall; neighbor = another eligible hall.
    scored, error = allocate(participants, required_facilities, requested_range)
    if error:
        return [], error

    current = scored[-1]  # start from the worst eligible candidate to demonstrate improvement
    path = [current]
    improved = True

    while improved:
        improved = False
        for candidate in scored:
            if candidate["score"] < current["score"]:
                current = candidate
                path.append(current)
                improved = True
                break

    return path, None


def answer(query):
    q = query.lower().strip()

    # Interactive allocation intent.
    allocation_words = [
        "allocate", "recommend hall", "choose hall", "select hall",
        "which hall", "suitable hall", "hall for", "book hall"
    ]
    participants = parse_participants(query)
    facilities = parse_facilities(query)
    time_range = parse_time_range(query)

    if any(w in q for w in allocation_words) and participants is not None:
        scores, error = allocate(participants, facilities, time_range)
        if error:
            return error, []

        best = scores[0]
        lines = [
            f"🏫 **Recommended hall: {best['hall']}**",
            "",
            f"- Participants: **{participants}**",
            f"- Unused seats (U): **{best['unused']}**",
            f"- Missing facilities (M): **{best['missing']}**",
            f"- Booking clashes (Cl): **{best['clash']}**",
            f"- Distance: **{best['distance']} m**",
            f"- Heuristic score: **{best['score']}**",
            "",
            "The assignment treats this as a minimization problem, so the lower heuristic value is preferred.",
            "",
            "**Formula:** H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)"
        ]
        return "\n".join(lines), scores

    # Common direct questions can be answered cleanly while still using the PDF as source.
    if "formula" in q or "heuristic function" in q:
        return (
            "**Heuristic function from the assignment:**\n\n"
            "H(A) = W1(U) + W2(M) + W3(Cl) + W4(Dist)\n\n"
            "For the example weights:\n"
            "**H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)**\n\n"
            "U = unused seats, M = missing facilities, Cl = booking clash penalty, "
            "and Dist = distance from the organizing department. Lower is better."
        ), []

    if "hill climbing" in q:
        return (
            "The assignment uses Hill Climbing as a simple search approach. "
            "It starts with an allocation, calculates its heuristic value, creates a "
            "neighboring allocation by changing a hall, and accepts the neighbor when "
            "its heuristic value is lower. It stops when no better neighboring allocation exists."
        ), []

    if "limitation" in q or "limitations" in q:
        return (
            "The assignment identifies these limitations: the result depends on the selected "
            "weights; participant counts can change; a low score can hide an important missing "
            "facility; Hill Climbing can reach a local optimum; and hall/booking data must be updated."
        ), []

    if "objective" in q or "objectives" in q:
        return (
            "The objectives are to find a suitable hall, reduce empty seats, provide required "
            "facilities, avoid booking clashes, reduce travel distance, and provide a good allocation "
            "without checking every possible allocation."
        ), []

    if "input" in q or "inputs" in q:
        return (
            "The system can use event name/date/time, expected participants, required facilities, "
            "hall capacity, hall facilities, existing bookings, and distance from the organizing department."
        ), []

    # PDF retrieval fallback.
    retrieved = retrieve(query, top_k=3)
    if not retrieved:
        return (
            "I couldn't find a reliable answer in the provided assignment. "
            "Try asking about the heuristic function, inputs, objectives, Hall A/Hall B example, "
            "Hill Climbing, pseudocode, advantages, limitations, or improvements."
        ), []

    context = "\n\n".join(chunk for chunk, score in retrieved)
    # Extract a concise answer from the retrieved material.
    return (
        "Based on the provided assignment:\n\n"
        + context[:3000]
        + "\n\nIf you want, ask me to explain this in simpler words."
    ), retrieved


st.set_page_config(
    page_title="Seminar Hall Allocation AI Assistant",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Seminar Hall Allocation AI Assistant")
st.caption("A PDF-based college chatbot + heuristic hall allocation system")

with st.sidebar:
    st.header("About the project")
    st.write(
        "This chatbot is built from the seminar hall allocation assignment. "
        "It can answer assignment questions and calculate a heuristic score for sample halls."
    )
    st.divider()
    st.subheader("Heuristic weights")
    st.code("H(A) = 2(U) + 20(M) + 100(Cl) + 1(Dist)")
    st.caption("Lower score = more suitable allocation")
    st.divider()
    st.subheader("Try")
    st.write("• What is the heuristic function?")
    st.write("• Explain Hill Climbing")
    st.write("• Allocate a hall for 120 participants with projector, mic and AC")
    st.write("• What are the limitations?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("scores"):
            st.dataframe(message["scores"], use_container_width=True)

prompt = st.chat_input("Ask about the assignment or request a hall allocation...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response, scores = answer(prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "scores": scores
    })

    with st.chat_message("assistant"):
        st.markdown(response)
        if scores:
            st.dataframe(scores, use_container_width=True)

st.divider()
st.caption("Knowledge source: AI_Seminar_Hall_Allocation.pdf | Educational prototype")
