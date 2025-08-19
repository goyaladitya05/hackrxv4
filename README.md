# HackRx LLM Query-Retrieval System

A robust, production-ready backend for document question-answering using state-of-the-art Large Language Models (LLMs) and vector search. Upload or link documents (PDF, DOCX, TXT, CSV, XLSX, EML, images, etc.) and query them with natural language questions. The system extracts, splits, embeds, retrieves, and answers using OpenAI GPT-4o, Gemini 2.5 Pro, and more.

## Features
- **Multi-format Document Support:** PDF, DOCX, TXT, CSV, XLSX, EML, images (with OCR), and more
- **FastAPI Backend:** High-performance, async API with secure token-based authentication
- **Advanced Embeddings:** GPU-accelerated SentenceTransformers (PyTorch)
- **Vector Search:** Efficient context retrieval using FAISS
- **LLM Integration:** OpenAI GPT-4o, Gemini 2.5 Pro, with fallback logic
- **LangChain Ecosystem:** Text splitting, tracing, and prompt management
- **Robust Logging & Health Checks:** For production reliability

## How It Works
1. **Document Ingestion:** Upload or provide a URL to your document
2. **Extraction:** Text and images are extracted (with OCR for scanned files)
3. **Chunking:** Text is split into overlapping chunks for context
4. **Embedding:** Chunks and questions are embedded using SentenceTransformers
5. **Retrieval:** FAISS vector search finds the most relevant context
6. **Answer Generation:** LLMs generate concise, context-grounded answers

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set environment variables:**
   - Copy `.env.example` to `.env` and fill in your API keys (OpenAI, Gemini, etc.)
3. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **API Endpoints:**
   - `POST /api/v1/hackrx/run` — Submit document URL and questions
   - `GET /api/v1/health` — Health check

## Technologies Used
- Python, FastAPI, Pydantic
- PyTorch, SentenceTransformers
- FAISS (vector search)
- OpenAI API, Google Gemini API
- LangChain, LangSmith
- pdfplumber, python-docx, BeautifulSoup, pytesseract (OCR)

## Project Structure
```
app/
  main.py        # FastAPI entrypoint
  routes.py      # API endpoints
  model.py       # Pydantic models
  config.py      # Settings
  auth.py        # Auth logic
  tracing.py     # LangSmith tracing
  services/
    loader.py    # Document loaders
    splitter.py  # Text chunking
    embeddings.py# Embedding logic
    vectorstore.py# FAISS logic
    llm_client.py# LLM integration
    logger.py    # Logging
    logic.py     # Retrieval & reasoning
```

## License
Apache 2.0

## Images Included

The repository includes the following images located in the `images/` folder:

![Screenshot (47)](images/Screenshot%20(47).png)
![Screenshot (49)](images/Screenshot%20(49).png)
![Screenshot (50)](images/Screenshot%20(50).png)
![Screenshot (51)](images/Screenshot%20(51).png)

<!-- Add screenshots, architecture diagrams, or demo GIFs here -->
