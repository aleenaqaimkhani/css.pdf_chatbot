import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
from PyPDF2 import PdfReader

# ---------------------- Minimal spacing CSS ----------------------
st.set_page_config(page_title="📘 CSS MindMap", layout="centered")

# Reduce bottom spacing so text input moves up
st.markdown("""
<style>
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- Setup ----------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found in environment. Please set it in your .env file.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash-8b")

# ---------------------- Title & Description ----------------------
st.title("📘 CSS MindMap")
st.markdown("""
This chatbot is specially designed for CSS (Central Superior Services) aspirants.  
You can ask any question related to CSS, and it will provide accurate answers based on authentic PDF content.  
Whether you need a brief explanation or a detailed one, simply choose your preference and language.  
It’s a smart, interactive assistant to help guide you through your CSS preparation with both text and audio support.
""")

# ---------------------- Sidebar ----------------------
language_codes = {
    "English": "en",
    "Urdu": "ur",
    "Hindi": "hi",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "Chinese": "zh-CN",
    "Russian": "ru",
    "Turkish": "tr"
}

with st.sidebar:
    selected_language = st.selectbox("Select Answer Language", list(language_codes.keys()), index=0)
    answer_length = st.radio("Answer Length", ["Short", "Detailed"], index=0)

# ---------------------- Helpers ----------------------
@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def text_to_speech(text: str, lang_code: str):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def build_prompt(messages, pdf_text, answer_length, selected_language):
    history_snippets = []
    for m in messages:
        if m["role"] == "user":
            history_snippets.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            history_snippets.append(f"Assistant: {m['content']}")

    conversation_text = "\n".join(history_snippets)

    prompt = f"""
You are an educational assistant. Use the following textbook content to answer.

Textbook Content:
\"\"\"{pdf_text}\"\"\"

Conversation so far:
{conversation_text}

Now, give a {answer_length.lower()} explanation in {selected_language} for the last user question.
If the question is unrelated, respond politely.
"""
    return prompt

# ---------------------- Load PDF ----------------------
pdf_text = extract_pdf_text("converted_text.pdf")

# ---------------------- Session State ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# ---------------------- Display Past Messages ----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio_bytes" in msg and msg["audio_bytes"]:
            st.audio(msg["audio_bytes"], format="audio/mp3")

# ---------------------- User Input ----------------------
user_input = st.chat_input("Ask anything about CSS…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build prompt
    prompt = build_prompt(
        st.session_state.messages,
        pdf_text=pdf_text,
        answer_length=answer_length,
        selected_language=selected_language
    )

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

        st.markdown(answer)

        # TTS
        lang_code = language_codes.get(selected_language, "en")
        audio_file = text_to_speech(answer, lang_code)
        audio_bytes = None
        if audio_file:
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "audio_bytes": audio_bytes
        })
        st.session_state.last_answer = answer

# ---------------------- Download Last Answer ----------------------
if st.session_state.last_answer:
    st.download_button(
        label="📩 Download last answer (text)",
        data=st.session_state.last_answer,
        file_name="css_answer.txt",
        mime="text/plain"
    )
