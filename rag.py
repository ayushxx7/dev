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
st.title("RAG Chatbot for YouTube Videos")
st.write("Ask questions based on the contents of the database.")

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
    st.info("Retrieving context from the content...")
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"""
    st.info("Querying Gemini...")
    response = model.generate_content(prompt)
    answer = response.text if response else "No response from Gemini."
    st.markdown(f"**Answer:** {answer}")
