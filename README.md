# Local Ollama Strict RAG Assistant

A fast, private, and fully offline Retrieval-Augmented Generation (RAG) web application built with **Streamlit** and **Ollama**. It runs entirely on your local machine to ensure 100% data privacy with zero cloud costs.

## Features
* **100% Offline & Private:** Powered locally by Ollama (`llama3.2` and `nomic-embed-text`).
* **Persistent Vector Caching:** Automatically caches document embeddings locally to disk (`vector_cache/`), allowing instant reloading on subsequent sessions.
* **Interactive UI:** Built with Streamlit for clean document management and quick question-answering.

## Prerequisites
Make sure you have the following installed on your machine:
1. **Python** (3.8 or higher)
2. **Ollama** installed and running locally ([Download Ollama](https://ollama.com/))

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tsamajdar/localrag.git](https://github.com/tsamajdar/localrag.git)
   cd your-repo-name