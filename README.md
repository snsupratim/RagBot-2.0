# 🧠 Modular RAG PDF Chatbot with FastAPI, ChromaDB & Streamlit

## 🎥 Watch the Tutorial

[![Watch the video](assets/ragbot2.0.png)](https://youtu.be/TxtK6NUUklQ)

This project is a modular **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and chat with an AI assistant that answers queries based on the document content. It features a microservice architecture with a decoupled **FastAPI backend** and **Streamlit frontend**, using **ChromaDB** as the vector store and **Groq's LLaMA3 model** as the LLM.

---

## 📂 Project Structure

```
ragbot2.0/
├── client/         # Streamlit Frontend
│   |──components/
|   |  |──chatUI.py
|   |  |──history_download.py
|   |  |──upload.py
|   |──utils/
|   |  |──api.py
|   |──app.py
|   |──config.py
├── server/         # FastAPI Backend
│   ├── chroma_store/ ....after run
|   |──modules/
│      ├── load_vectorestore.py
│      ├── llm.py
│      ├── pdf_handler.py
│      ├── query_handlers.py
|   |──uploaded_pdfs/ ....after run
│   ├── logger.py
│   └── main.py
└── README.md
```

---

## ✨ Features

- 📄 Upload and parse PDFs
- 🧠 Embed document chunks with HuggingFace embeddings
- 💂️ Store embeddings in ChromaDB
- 💬 Query documents using LLaMA3 via Groq
- 🌍 Microservice architecture (Streamlit client + FastAPI server)

---

## 🎓 How RAG Works

Retrieval-Augmented Generation (RAG) enhances LLMs by injecting external knowledge. Instead of relying solely on pre-trained data, the model retrieves relevant information from a vector database (like ChromaDB) and uses it to generate accurate, context-aware responses.

---

## 📊 Application Diagram

📄 [Download the Full Architecture PDF](assets/ragbot2.0.pdf)

---

## 🚀 Getting Started Locally

### 1. Clone the Repository

```bash
git clone https://github.com/snsupratim/ragbot.git
cd ragbot
```

### 2. Setup the Backend (FastAPI)

```bash
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your Groq API Key
echo "GROQ_API_KEY=your_key_here" > .env

# Run the FastAPI server
uvicorn main:app --reload
```

### 3. Setup the Frontend (Streamlit)

```bash
cd ../client
pip install -r requirements.txt  # if you use a separate venv for client
streamlit run app.py
```

---

## 🌐 API Endpoints (FastAPI)

- `POST /upload_pdfs/` — Upload PDFs and build vectorstore
- `POST /ask/` — Send a query and receive answers

Testable via Postman or directly from the Streamlit frontend.

---

## 🚧 TODO

- [ ] Add authentication for endpoints
- [ ] Dockerize the project
- [ ] Add support for more file types

---

## 🌟 Credits

- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/)
- [Streamlit](https://streamlit.io/)

---

## ✉️ Contact

For questions or suggestions, open an issue or contact at [snsupratim@gmail.com]

---

> Happy Building RAGbots! 🚀
