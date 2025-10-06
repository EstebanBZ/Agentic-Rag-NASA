# Agentic-Rag-NASA
Hesoyam team - NASA hackathon 2025 - Agentic-Rag for Biosceince investigation


# NASA Bio-RAG — Agentic RAG + Graph + PDF Dashboard

## Overview

This project integrates a **Retrieval-Augmented Generation (RAG)** pipeline with an **interactive graph visualization** and a **PDF document search interface**. It allows users to:

* Ask questions about NASA biosciences data through a chatbot.
* Visualize relationships between scientific publications as a network graph.
* Search, view, and open related PDF documents directly from the dashboard.

The system is built using **FastAPI**, **LangChain**, **FAISS**, **Plotly**, and **HuggingFace embeddings**.

---

## Features

### 🧠 Agentic RAG

* Integrates a LangChain RAG pipeline powered by **Azure OpenAI** and **FAISS vector storage**.
* Uses `HuggingFaceEmbeddings` (model: `BAAI/bge-small-en-v1.5`).

### 🌐 Interactive Graph

* Automatically builds a **TF-IDF concept graph** from `.md` documents located in the `./extracted_md` folder.
* Visualizes publication relationships using **Plotly**.
* Allows highlighting of specific concepts by keyword.

### 📄 PDF Search and Viewer

* Search PDFs by filename (case-insensitive) inside a local `pdfs` folder.
* Preview selected PDFs within the dashboard.
* Option to open files in a new tab.

### 💬 Chat Interface

* Simple two-way chat between user and the AI assistant.
* Messages are rendered in Markdown with syntax highlighting.

---

## Project Structure

```
project_root/
│
├── main.py                # FastAPI backend
├── graph.py               # Graph building logic
├── index.html             # Frontend UI (chat, graph, PDF viewer)
├── .env                   # Environment variables (ignored by git)
├── .gitignore             # Git ignore rules
├── vector_db/             # FAISS vector database
├── extracted_md/          # Folder with Markdown (.md) documents
└── pdfs/                  # Folder with related PDFs
```

---

## Environment Variables (`.env`)

Create a `.env` file in the project root with the following keys:

```
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
AZURE_OPENAI_API_KEY=<your_api_key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-12-01-preview

```

> ⚠️ Do **not** upload your `.env` file to GitHub — it contains sensitive credentials.

---

## Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/nasa-bio-rag.git
cd nasa-bio-rag
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` file

Add your Azure OpenAI credentials and paths as shown above.

### 5️⃣ Place your data

* Place your `.md` files in the folder `extracted_md/`
* Place your `.pdf` files in the folder `pdfs/`

---

## Run the Application

### 🧩 Development Server

Start the FastAPI app using **Uvicorn** with auto-reload:

```bash
uvicorn main:app --reload
```

Then open your browser at:

👉 [http://localhost:8000/](http://localhost:8000/)

### ⚙️ Custom Options

* Change port: `uvicorn main:app --reload --port 8001`
* Expose on local network: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`

---



## Example Usage

1. Open the dashboard at [http://localhost:8000/](http://localhost:8000/)
2. Ask questions in the left-side chat.
3. Highlight graph concepts in the top-right input.
4. Search and open PDF documents in the bottom-right section.

---
