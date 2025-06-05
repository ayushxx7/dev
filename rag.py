import streamlit as st
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import random


# Configure Gemini
genai.configure(api_key="AIzaSyCWWp87jq69qbFdC2hIvd1B7QgZf0QuS5U")
model = genai.GenerativeModel("gemini-1.5-pro")

# Load JSON data from a file
def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Extract text from JSON data
def extract_text_from_json(json_data):
    text = ""
    for video in json_data.get("videos", []):
        title = video.get("title", "")
        description = video.get("description", "")
        text += f"Title: {title}\nDescription: {description}\n\n"
    return text

# Create FAISS vector store
def create_faiss_vector_store(text, path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts([text], embedding=embeddings)
    vector_store.save_local(path)

# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

# Bollywood quotes/facts
BOLLYWOOD_QUOTES = [
    "“Bade bade deshon mein aisi choti choti baatein hoti rehti hain!” 🎬",
    "“Mogambo khush hua!” 😎",
    "“Picture abhi baaki hai mere dost!” 🍿",
    "“Don ko pakadna mushkil hi nahi, namumkin hai!” 🕶️",
    "“All is well!” 🤞",
    "“How’s the josh? High Sir!” 💥",
]

# Funky Bollywood loading quotes
BOLLYWOOD_LOADING_QUOTES = [
    "🎬 Loading... Picture abhi baaki hai mere dost!",
    "💃🕺 Loading... How’s the josh? High Sir!",
    "🍿 Loading... Mogambo khush hua!",
    "🎤 Loading... All is well!",
    "🕶️ Loading... Don ko pakadna mushkil hi nahi, namumkin hai!",
    "✨ Loading... Bade bade deshon mein aisi choti choti baatein hoti rehti hain!",
]

# Streamlit App
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://www.bollywoodhungama.com/wp-content/themes/bh-theme/images/logo.png' width='250'/>
        <h1 style='color: #e50914; font-family: "Comic Sans MS", cursive, sans-serif;'>Bollywood Hungama Chatbot 🎬✨</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Show a random Bollywood quote/fact (not loading quote)
st.markdown(
    f"<div style='text-align:center; color:#e91e63; font-size:22px; margin-bottom:10px;'>"
    f"💬 <i>{random.choice(BOLLYWOOD_QUOTES)}</i></div>",
    unsafe_allow_html=True
)

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

JSON_PATH = "youtube_videos.json"

# Load and process JSON data
try:
    st.info(random.choice(BOLLYWOOD_LOADING_QUOTES))
    json_data = load_json_data(JSON_PATH)
    text = extract_text_from_json(json_data)
except Exception as e:
    st.error(f"😱 Arre baba! Error loading JSON data: {e}")
    st.stop()

if not text.strip():
    st.error("😅 Kya karen, no usable content found in the JSON file!")
    st.stop()

create_faiss_vector_store(text)
st.info("🎬 Lights, Camera, Action! The chatbot is ready for your Bollywood questions!")
vector_store = load_faiss_vector_store()

question = st.text_input("🎤 Ask a question about YouTube videos:")

# Give your Bollywood bot a fun name!
BOT_NAME = "HungamaBot"

if question:
    # Build chat history string for retrieval
    retrieval_query = ""
    for entry in st.session_state.chat_history:
        retrieval_query += f"User: {entry['question']}\n{BOT_NAME}: {entry['answer']}\n"
    retrieval_query += f"User: {question}"

    st.info("🔎 Searching for your answer... Don ko pakadna mushkil hi nahi, namumkin hai!")
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(retrieval_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build chat history string for bot prompt
    history_str = ""
    for entry in st.session_state.chat_history:
        history_str += f"User: {entry['question']}\n{BOT_NAME}: {entry['answer']}\n"

    # Compose prompt with history
    prompt = (
        f"{history_str}"
        f"Context:\n{context}\n\n"
        f"User: {question}\n{BOT_NAME}:"
    )

    st.info(f"🤖 {BOT_NAME} is thinking... Picture abhi baaki hai mere dost!")
    response = model.generate_content(prompt)
    answer = response.text if response else "No response from HungamaBot."
    st.session_state.chat_history.append({"question": question, "answer": answer})
    st.success("🎉 Mogambo khush hua! Here's your Bollywood answer!")

# Display chat history with Bollywood style
if st.session_state.chat_history:
    st.markdown("### 📝 Chat History")
    for entry in st.session_state.chat_history[::-1]:
        st.markdown(
            f"<div style='background:#f3f7fa; border-radius:10px; padding:10px; margin-bottom:8px;'>"
            f"<b style='color:#1976d2;'>🧑‍🎤 You:</b> <span style='color:#333;'>{entry['question']}</span><br>"
            f"<b style='color:#388e3c;'>🤖 {BOT_NAME}:</b> <span style='color:#222;'>{entry['answer']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

# Funky Bollywood footer
st.markdown(
    """
    <hr>
    <div style='text-align:center; font-size:16px; color:#ff9800;'>
        Made with ❤️ for Bollywood fans | <a href='https://www.instagram.com/zeemusiccompany/' target='_blank'>Follow Zee Music on Instagram</a>
    </div>
    """,
    unsafe_allow_html=True
)
