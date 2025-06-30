# RAG Chatbot with LangChain and Gemini

This project implements a Retrieval-Augmented Generation (RAG) chatbot using [LangChain](https://www.langchain.com/), local embeddings with [ChromaDB](https://www.trychroma.com/), and an LLM (Gemini).

## Features

- Document ingestion and chunking with semantic embeddings
    - Articles are related to competencies for one's well-being
- Context-aware question answering
- Support for cloud-based LLM (Gemini)

---

## Setup

### 1. Clone the repository

```
git clone https://github.com/ylee862/RAG-Model.git
cd RAG-Model
```

### 2. Create a virtual environment
```
python3 -m venv modelvenv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Add .env File
```
GOOGLE_API_KEY=your_google_api_key
```

### 5. Run data indexing script
```
python3 indexing.py
```

### 6. Insert the prompt
```
python3 query_data.py "What is self-management?"
```

