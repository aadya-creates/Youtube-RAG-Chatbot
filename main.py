from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import os
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
link=input("Enter the YouTube video link: ")
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed = urlparse(url)

    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query)["v"][0]

    elif "youtu.be" in parsed.netloc:
        return parsed.path[1:]

    else:
        raise ValueError("Invalid YouTube URL")

Vid_id=extract_video_id(link)
ytt=YouTubeTranscriptApi()
try:
    transcript=ytt.fetch(video_id=Vid_id, 
    languages=[
        "en",
        "en-US",
        "en-GB",
        "en-IN",
        "en-CA",
        "en-AU"
    ])
except Exception as e:
    print("Transcript unavailable")
    print(e)
    exit()


document = []

for chunk in transcript:
    doc = Document(
        page_content=chunk.text,
        metadata={
            "timestamp": chunk.start
        }
    )
    document.append(doc)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100)

chunks=splitter.split_documents(document)

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

vs=FAISS.from_documents(chunks, embedding_model)

retriever=vs.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)
model=ChatMistralAI(model="mistral-small-latest", temperature=0.3)

prompt_template = ChatPromptTemplate.from_template("""
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
prompt = ChatPromptTemplate.from_template(prompt_template)
chat_history=[]

while True:
    query=input("question (type 0 to exit): ")
    if query=="0":
        break
    docs=retriever.invoke(query)
    context=""
    for doc in docs:
        timestamp = int(doc.metadata["timestamp"])

        minutes = timestamp // 60
        seconds = timestamp % 60

        readable_time = f"{minutes}:{seconds:02d}"

        context += f"""Timestamp: {readable_time}

        Content:
        {doc.page_content}

        """

    history_str=""
    for chat in chat_history:
        history_str += f"User: {chat['user']}\n"
        history_str += f"AI: {chat['ai']}\n"
    final_prompt=prompt.invoke({
        "context": context,
        "question": query,
        "history": history_str
    })

    response=model.invoke(final_prompt)
    print("Answer: ", response.content)

    chat_history.append({
    "user": query,
    "ai": response.content
    })


