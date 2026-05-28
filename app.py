import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from urllib.parse import urlparse, parse_qs
import os

def extract_video_id(url):
    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]
    elif "youtu.be" in parsed.netloc:
        return parsed.path[1:]
    else:
        raise ValueError("Invalid YouTube URL")


# -------------------------
# Streamlit config
# -------------------------
st.set_page_config(page_title="YouTube RAG", layout="wide")
st.title("🎥 YouTube RAG Assistant")


# -------------------------
# Session state
# -------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------
# Sidebar (controls)
# -------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    link = st.text_input("YouTube URL")

    if st.button("Process Video") and link:

        vid_id = extract_video_id(link)

        ytt = YouTubeTranscriptApi()

        try:
            transcript = ytt.fetch(video_id=vid_id, languages=["en"])
        except Exception as e:
            st.error("Transcript not available")
            st.stop()

        docs = [
            Document(page_content=chunk.text, metadata={"timestamp": chunk.start})
            for chunk in transcript
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        split_docs = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(split_docs, embeddings)

        st.session_state.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3}
        )

        st.success("Video processed 🚀")


# -------------------------
# LLM + Prompt
# -------------------------
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
You are an AI tutor.

Your job:
- Answer the question using ONLY the context
- DO NOT copy sentences directly from context
- Always explain in your own words
- If context is messy, still summarize it clearly

Context:
{context}

Question:
{question}

Chat History:
{history}

Rules:
- Never repeat raw transcript lines
- Never start answer with timestamps or quotes
- Give a clean explanation first
- Mention timestamp only at the END like (00:33)
""")

if st.session_state.retriever:

    query = st.chat_input("Ask something about the video...")

    if query:

        docs = st.session_state.retriever.invoke(query)

        context = ""
        for doc in docs:
            ts = int(doc.metadata["timestamp"])
            mm = ts // 60
            ss = ts % 60
            time = f"{mm:02d}:{ss:02d}"

            context += f"[{time}] {doc.page_content}\n"

        # history formatting
        history = ""
        for chat in st.session_state.chat_history:
            history += f"User: {chat['user']}\nAI: {chat['ai']}\n"

        final_prompt = prompt.invoke({
            "context": context,
            "question": query,
            "history": history
        })

        response = model.invoke(final_prompt)

        # save memory
        st.session_state.chat_history.append({
            "user": query,
            "ai": response.content
        })


for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["user"])

    with st.chat_message("assistant"):
        st.write(chat["ai"])