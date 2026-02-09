# 🧠 AI-Powered E-Commerce Assistant  
**A production-style LLM system showcasing semantic routing, RAG, SQL reasoning, and multi-agent orchestration**

---

## 🚀 Project Overview

The **AI-Powered E-Commerce Assistant** is a real-world inspired **LLM-driven application** designed to demonstrate how modern AI systems go beyond simple chatbots.

Instead of using a single model for everything, this system intelligently:
- Understands **user intent**
- Routes queries to **specialized pipelines**
- Combines **LLMs, vector databases, and structured SQL data**
- Produces grounded, non-hallucinated responses

This project reflects **how AI assistants are built in production environments**.

---

## 🧠 Core AI Concepts Demonstrated

✅ **Semantic Routing with Embeddings**  
User queries are routed based on semantic similarity, not keywords, enabling robust intent detection.

✅ **Retrieval-Augmented Generation (RAG)**  
FAQs and general knowledge are answered using vector search + LLMs to prevent hallucinations.

✅ **Natural Language → SQL Reasoning**  
LLMs generate safe SQL queries from natural language and reason over structured product data.

✅ **Context-Aware Conversations**  
Conversation summaries and recent context are used to maintain continuity across turns.

✅ **Fallback & Clarification Agent**  
Gracefully handles unsupported or ambiguous queries instead of hallucinating.

---

## 🔬 AI / ML Tech Stack

- **LLM Inference**: Groq API  
- **Primary Reasoning Model**: `llama-3.3-70b-versatile`  
- **Routing**: Semantic Router (embedding-based intent classification)  
- **Embeddings**: Sentence Transformers (MiniLM)  
- **Vector Database**: ChromaDB  
- **Structured Data**: SQLite  
- **Frontend**: Streamlit  
- **Language**: Python  

---

## 🧪 Example Queries

- *“Show me running shoes under ₹3000”*  
- *“Are there any Puma shoes on discount?”*  
- *“What is the refund policy for defective products?”*  
- *“Can you compare two products for daily use?”*  

Each query is **routed to a different AI pipeline**.

---

## 🧠 Why This Is Not a Simple Chatbot

Unlike basic LLM chat apps, this system:

- ❌ Does not rely on a single prompt  
- ❌ Does not hallucinate FAQ answers  
- ❌ Does not hardcode logic  

Instead, it:
- Uses **multiple AI subsystems**
- Separates **retrieval, reasoning, and response generation**
- Mimics **real AI assistant architectures used in industry**

---

## 🔐 Engineering & Safety Considerations

- SQL execution guarded against unsafe queries  
- No hallucinated answers for RAG pipelines  
- Session-level isolation for multi-user safety  
- Modular design for easy extension  

---

## 🚧 Future Enhancements

- 🔄 Model switching based on query complexity  
- 📉 Token-optimized prompt pipelines  
- 🧪 Automated evaluation & response scoring  
- 🔐 User personalization & memory persistence  
- 📊 Recommendation ranking using ML  

---

## 👨‍💻 About the Developer

I am a **Computer Science & Engineering student** with strong interest in:

- Generative AI & LLM systems  
- AI system design & orchestration  
- Retrieval-Augmented Generation  
- Data-driven and production-style AI applications  

This project reflects my focus on **building AI systems, not just calling APIs**.

---

## 📬 Feedback & Contributions

Suggestions and improvements are welcome!  
Feel free to fork, star ⭐, or raise an issue.

---

