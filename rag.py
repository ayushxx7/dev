import streamlit as st
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


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

# Streamlit App
st.markdown(
    """
    <div style='text-align: center;'>
        <img src='https://www.bollywoodhungama.com/wp-content/themes/bh-theme/images/logo.png' width='250'/>
        <h1 style='color: #e50914; font-family: "Comic Sans MS", cursive, sans-serif;'>Bollywood Hungama Chatbot 🎬✨</h1>
        <p style='font-size: 20px; color: #ff9800;'>Lights, Camera, Action! Ask anything about your favorite Bollywood videos and let the Hungama begin! 💃🕺</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.balloons()

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Remove the file uploader and use a fixed path
JSON_PATH = "youtube_videos.json"

# Load and process JSON data
try:
    st.info("Loading vector database...")
    json_data = load_json_data(JSON_PATH)
    text = extract_text_from_json(json_data)
except Exception as e:
    st.error(f"Error loading JSON data: {e}")
    st.stop()

if not text.strip():
    st.error("No usable content found in the JSON file.")
    st.stop()

create_faiss_vector_store(text)
st.success("Chatbot is ready!")
vector_store = load_faiss_vector_store()

question = st.text_input("Ask a question about YouTube videos:")

if question:
    # Build chat history string for retrieval
    retrieval_query = ""
    for entry in st.session_state.chat_history:
        retrieval_query += f"User: {entry['question']}\nBot: {entry['answer']}\n"
    retrieval_query += f"User: {question}"

    st.info("Retrieving context from the content...")
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(retrieval_query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build chat history string for Gemini prompt
    history_str = ""
    for entry in st.session_state.chat_history:
        history_str += f"User: {entry['question']}\nBot: {entry['answer']}\n"

    # Compose prompt with history
    prompt = (
        f"{history_str}"
        f"Context:\n{context}\n\n"
        f"User: {question}\nBot:"
    )

    st.info("Querying Gemini...")
    response = model.generate_content(prompt)
    answer = response.text if response else "No response from Gemini."
    st.session_state.chat_history.append({"question": question, "answer": answer})

# Display chat history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    for entry in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {entry['question']}")
        st.markdown(f"**Bot:** {entry['answer']}")
