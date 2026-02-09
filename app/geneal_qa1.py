import pandas as pd
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

# ---------------- ENV ----------------
load_dotenv()

# ---------------- PATHS & CLIENTS ----------------
general_qa_path = Path(__file__).parent / "resources/ecommerce_chatbot_qna.csv"
chroma_client = chromadb.Client()
COLLECTION_NAME = "general_qa_client"

groq = Groq()

# ---------------- EMBEDDINGS ----------------
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ---------------- MEMORY CONFIG ----------------
conversation_summary = ""
recent_messages = []

MAX_RECENT_MESSAGES = 4
SUMMARY_TRIGGER = 8

# ---------------- INGESTION ----------------
def general_data_ingest(path):
    existing_collections = [c.name for c in chroma_client.list_collections()]

    if COLLECTION_NAME not in existing_collections:
        print("📥 Loading data into ChromaDB...")

        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )

        df = pd.read_csv(path)
        documents = df["question"].tolist()
        metadatas = [{"answer": ans} for ans in df["answer"].tolist()]
        ids = [f"id_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        print("✅ Data loaded successfully")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists")

# ---------------- RETRIEVAL ----------------
def query_relevant_answ(query):
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    result = collection.query(
        query_texts=[query],
        n_results=2,
    )

    return result

# ---------------- SUMMARY ----------------
def summarize_conversation(messages):
    conversation_text = "\n".join(messages)

    prompt = f"""
Summarize the following conversation.
Keep only key facts and user intent.

Conversation:
{conversation_text}
"""

    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content

# ---------------- ANSWER GENERATION ----------------
def generate_answer(query, context, summary="", chat_history=""):
    prompt = f"""
    You are a helpful e-commerce assistant.
    
    Conversation summary:
    {summary}
    
    Recent conversation:
    {chat_history}
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer strictly based on the context.
    If the answer is not found, say "I don't know".
    """

    completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content

# ---------------- MAIN QA CHAIN ----------------
def general_qa_chain(query):
    global conversation_summary, recent_messages

    # ---- RETRIEVE ----
    retrieved = query_relevant_answ(query)
    context = " ".join(
        meta.get("answer") for meta in retrieved["metadatas"][0]
    )

    # ---- GENERATE ----
    answer = generate_answer(
        query=query,
        context=context,
        summary=conversation_summary,
        chat_history="\n".join(recent_messages),
    )

    # ---- UPDATE MEMORY ----
    recent_messages.append(f"User: {query}")
    recent_messages.append(f"Assistant: {answer}")

    if len(recent_messages) > SUMMARY_TRIGGER:
        new_summary = summarize_conversation(recent_messages)
        conversation_summary += "\n" + new_summary
        recent_messages = recent_messages[-MAX_RECENT_MESSAGES * 2 :]

    return answer

# ---------------- TEST ----------------
if __name__ == "__main__":
    general_data_ingest(general_qa_path)

    q1 = "what is your role"
    print("🤖:", general_qa_chain(q1))

    q2 = "what skills are required?"
    print("🤖:", general_qa_chain(q2))
