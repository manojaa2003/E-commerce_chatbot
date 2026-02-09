import pandas as pd
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

general_qa_path = Path(__file__).parent/"resources/ecommerce_chatbot_qna.csv"
chroma_client = chromadb.Client()
collections_name = "general_qa_client"
groq = Groq()


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
)

convo_summary = ""
recent_chats = []

Max_Chat_Size = 4
Summary_trigger = 8

def general_data_ingest(path):
    if collections_name not in [collection.name for collection in chroma_client.list_collections()]:
        print("Loading data to Chroma database")
        collections = chroma_client.get_or_create_collection(
            name=collections_name,
            embedding_function=ef,
        )

        df = pd.read_csv(general_qa_path)
        docs = df['question'].tolist()
        metadata = [{"answer" : data} for data in df['answer'].tolist()]

        ids = [f"id_{i}" for i in range(len(docs))]

        collections.add(
            documents = docs,
            metadatas= metadata,
            ids=ids,
        )
        print("Loaded data successfully")
    else:
        print(f"Collection -> {collections_name} already exist")

def query_relevant_answ(query):
    collection = chroma_client.get_collection(
        name=collections_name,
        embedding_function=ef,
    )

    result = collection.query(
        query_texts=query,
        n_results=2,
    )
    return result

def summary_generation(recent_mgs):
    prompt = f'''
    generate the summary for this below conversations.
    keep only key facts and user intent:
    {recent_mgs}
    '''
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def general_qa_chain(query):
    global convo_summary,recent_chats
    queried_answers = query_relevant_answ(query)
    context = " ".join(answ.get('answer') for answ in queried_answers['metadatas'][0])
    answer = generate_answer(
        query,
        context,
        summary=convo_summary,
        chat_history = "\n".join(recent_chats)
    )

    recent_chats.append(f"user : {query}")
    recent_chats.append(f"assistant : {answer}")

    if len(recent_chats) > Summary_trigger:
        summary = summary_generation(recent_chats)
        convo_summary = convo_summary + "\n" + summary
        recent_chats = recent_chats[-Max_Chat_Size*2:]

    return answer

def generate_answer(query,context,summary,chat_history):
    prompt = f'''
    Given the following context, question, summary of previous chats and chat history , generate answer based on these elements only.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.
    
    summary of previous convertion:
    {summary}
    
    chat history:
    {chat_history}
    
    Question:
    {query}
    
    Context:
    {context}
    '''
    chat_completion = groq.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=os.environ['GROQ_MODEL'],
    )

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    general_data_ingest(general_qa_path)
    query1 = "what is your role"
    print(general_qa_chain(query1))

    query2 = "what was my previous conversation"
    print(general_qa_chain(query2))
