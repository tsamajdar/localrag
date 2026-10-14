# ==========================================
# 1. IMPORTS & CONFIGURATION BLOCK
# Purpose: Imports Streamlit for UI, ChromaDB for vectors, 
# Ollama for LLMs, SQLite for relational metadata storage, and utilities.
# ==========================================
import hashlib
import os
import re
import sqlite3
import chromadb
from duckduckgo_search import DDGS
import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
DB_FILE = os.path.join(CHROMA_DIR, "app_metadata.db")

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

st.set_page_config(
    page_title="Local Ollama Chroma RAG Assistant", page_icon="🤖", layout="wide"
)

# ==========================================
# 2. CUSTOM UI STYLING BLOCK
# Purpose: Injects CSS styles for a modern Claude/Gemini-style 
# typing/loading bouncing-dots animation.
# ==========================================
st.markdown(
    """
    <style>
    .typing-loader {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 0;
    }
    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #4f46e5;
        border-radius: 50%;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    .typing-dot:nth-child(3) { animation-delay: 0s; }

    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 3. SQLITE DATABASE INITIALIZATION & HELPERS
# Purpose: Sets up relational tables for projects, settings, 
# and persistent per-project chat history.
# ==========================================
def init_db():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "CREATE TABLE IF NOT EXISTS projects (name TEXT PRIMARY KEY)"
  )
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_settings (
            project_name TEXT PRIMARY KEY,
            current_kb TEXT,
            chunk_size INTEGER,
            overlap INTEGER,
            allow_fallback INTEGER,
            response_style TEXT
        )
    """)
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT,
            role TEXT,
            content TEXT
        )
    """)
  conn.commit()
  conn.close()


init_db()


def db_get_projects():
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("SELECT name FROM projects")
  rows = cursor.fetchall()
  conn.close()
  if rows:
    return [row[0] for row in rows]
  return ["Default Project"]


def db_add_project(name):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute("INSERT OR IGNORE INTO projects (name) VALUES (?)", (name,))
  conn.commit()
  conn.close()


def db_get_settings(project_name):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "SELECT current_kb, chunk_size, overlap, allow_fallback, response_style"
      " FROM project_settings WHERE project_name = ?",
      (project_name,),
  )
  row = cursor.fetchone()
  conn.close()
  if row:
    return {
        "current_kb": row[0],
        "chunk_size": row[1] if row[1] else 350,
        "overlap": row[2] if row[2] else 40,
        "allow_fallback": bool(row[3]),
        "response_style": row[4] if row[4] else "Standard Detailed",
    }
  return {
      "current_kb": None,
      "chunk_size": 350,
      "overlap": 40,
      "allow_fallback": False,
      "response_style": "Standard Detailed",
  }


def db_save_settings(
    project_name,
    current_kb,
    chunk_size,
    overlap,
    allow_fallback,
    response_style,
):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO project_settings (project_name, current_kb, chunk_size, overlap, allow_fallback, response_style)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_name) DO UPDATE SET
            current_kb=excluded.current_kb,
            chunk_size=excluded.chunk_size,
            overlap=excluded.overlap,
            allow_fallback=excluded.allow_fallback,
            response_style=excluded.response_style
    """,
      (
          project_name,
          current_kb,
          chunk_size,
          overlap,
          int(allow_fallback),
          response_style,
      ),
  )
  conn.commit()
  conn.close()


def db_get_chat_history(project_name):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "SELECT role, content FROM chat_messages WHERE project_name = ?",
      (project_name,),
  )
  rows = cursor.fetchall()
  conn.close()
  return [{"role": row[0], "content": row[1]} for row in rows]


def db_add_chat_message(project_name, role, content):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "INSERT INTO chat_messages (project_name, role, content) VALUES (?, ?, ?)",
      (project_name, role, content),
  )
  conn.commit()
  conn.close()


def db_clear_chat_history(project_name):
  conn = sqlite3.connect(DB_FILE)
  cursor = conn.cursor()
  cursor.execute(
      "DELETE FROM chat_messages WHERE project_name = ?", (project_name,)
  )
  conn.commit()
  conn.close()


if not db_get_projects():
  db_add_project("Default Project")


# ==========================================
# 4. TEXT CLEANING & WEB SEARCH UTILITIES
# ==========================================
def clean_text(text: str) -> str:
  if not text:
    return ""
  text = re.sub(r"\n\s*\n", "\n\n", text)
  text = re.sub(r"[ \t]+", " ", text)
  return text.strip()


def perform_web_search(query: str, max_results=3) -> str:
  try:
    with DDGS() as ddgs:
      results = [r for r in ddgs.text(query, max_results=max_results)]
      if results:
        search_summary = ""
        for r in results:
          title = r.get("title", "")
          body = r.get("body", "")
          href = r.get("href", "")
          search_summary += (
              f"- Title: {title}\n  Snippet: {body}\n  URL: {href}\n\n"
          )
        return search_summary.strip()
  except Exception as e:
    return f"Web search error: {e}"
  return "No live search results found."


# ==========================================
# 5. CHROMA CLIENT INITIALIZATION
# ==========================================
@st.cache_resource
def get_chroma_client():
  return chromadb.PersistentClient(path=CHROMA_DIR)


chroma_client = get_chroma_client()


# ==========================================
# 6. VECTOR STORE CLASS
# ==========================================
class ChromaVectorStore:

  def __init__(self, project_name, filename, chunk_size=350, overlap=40):
    self.project_name = project_name
    self.filename = filename
    self.chunk_size = chunk_size
    self.overlap = overlap

    clean_proj = "".join(
        c if c.isalnum() or c in (".", "_", "-") else "_"
        for c in self.project_name
    ).strip("._-")
    if not clean_proj:
      clean_proj = "project"

    self.collection_name = f"{clean_proj}_cs{chunk_size}_ol{overlap}"
    self.collection = chroma_client.get_or_create_collection(
        name=self.collection_name
    )

  def is_indexed(self):
    return self.collection.count() > 0

  def add_document(self, text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    if not words:
      return

    existing_ids = self.collection.get()["ids"]
    if existing_ids:
      self.collection.delete(ids=existing_ids)

    chunks = []
    embeddings = []
    ids = []
    metadatas = []

    seen_chunk_hashes = set()
    chunk_indices = range(0, len(words), self.chunk_size - self.overlap)

    failed_chunks = 0
    for idx, i in enumerate(chunk_indices):
      chunk = " ".join(words[i : i + self.chunk_size])

      if len(chunk.strip()) > 30:
        chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
        if chunk_hash in seen_chunk_hashes:
          continue
        seen_chunk_hashes.add(chunk_hash)

        try:
          response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
          vector = response["embedding"]

          chunks.append(chunk)
          embeddings.append(vector)
          ids.append(f"chunk_{len(chunks)}")
          metadatas.append(
              {
                  "project": self.project_name,
                  "source": self.filename,
                  "chunk_index": str(len(chunks)),
              }
          )
        except Exception as e:
          failed_chunks += 1
          print(f"Embedding failed for chunk {idx}: {e}")

    if failed_chunks > 0:
      st.warning(
          f"⚠️ Completed with warnings: {failed_chunks} chunks failed to embed"
          " due to size or timeout."
      )

    if chunks:
      self.collection.add(
          documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas
      )

  def search(self, query, top_k=6):
    if not self.is_indexed():
      return []

    try:
      res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
      query_vector = res["embedding"]

      results = self.collection.query(
          query_embeddings=[query_vector], n_results=top_k
      )

      if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    except Exception as e:
      st.error(f"Error querying Chroma DB: {e}")

    return []


# ==========================================
# 7. STREAMLIT SESSION STATE & VECTOR RESTORATION
# ==========================================
if "current_project" not in st.session_state:
  projects_list = db_get_projects()
  st.session_state.current_project = (
      projects_list[0] if projects_list else "Default Project"
  )

if "vector_store" not in st.session_state:
  st.session_state.vector_store = None
if "doc_loaded" not in st.session_state:
  st.session_state.doc_loaded = False

proj_settings = db_get_settings(st.session_state.current_project)
if "current_kb" not in st.session_state:
  st.session_state.current_kb = proj_settings["current_kb"]

if (
    "cache_selector" in st.session_state
    and st.session_state.cache_selector != "-- Choose from cache --"
):
  st.session_state.current_kb = st.session_state.cache_selector

if st.session_state.current_kb and not st.session_state.vector_store:
  try:
    if "_cs" in st.session_state.current_kb and "_ol" in st.session_state.current_kb:
      parts = st.session_state.current_kb.rsplit("_cs", 1)
      params = parts[1].split("_ol")
      c_size = int(params[0])
      o_size = int(params[1])
    else:
      c_size = proj_settings["chunk_size"]
      o_size = proj_settings["overlap"]

    store = ChromaVectorStore(
        project_name=st.session_state.current_project,
        filename=st.session_state.current_kb,
        chunk_size=c_size,
        overlap=o_size,
    )
    if store.is_indexed():
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
  except Exception:
    pass

# ==========================================
# 8. MAIN UI HEADER
# ==========================================
st.title("🤖 Local Ollama Chroma RAG Assistant")
st.markdown(
    f"Ask questions below. Powered by Chroma DB, SQLite metadata, and Ollama"
    f" (`{GENERATION_MODEL}`)."
)


# ==========================================
# 9. SIDEBAR: PROJECT WORKSPACE & KNOWLEDGE BASE MANAGEMENT
# ==========================================
st.sidebar.title("🗂️ Project Workspace")


def handle_project_change():
  st.session_state.current_project = st.session_state.project_selector
  new_settings = db_get_settings(st.session_state.current_project)
  st.session_state.current_kb = new_settings["current_kb"]
  st.session_state.vector_store = None
  st.session_state.doc_loaded = False

  if st.session_state.current_kb:
    try:
      if "_cs" in st.session_state.current_kb and "_ol" in st.session_state.current_kb:
        parts = st.session_state.current_kb.rsplit("_cs", 1)
        params = parts[1].split("_ol")
        c_size = int(params[0])
        o_size = int(params[1])
      else:
        c_size = new_settings["chunk_size"]
        o_size = new_settings["overlap"]

      store = ChromaVectorStore(
          project_name=st.session_state.current_project,
          filename=st.session_state.current_kb,
          chunk_size=c_size,
          overlap=o_size,
      )
      if store.is_indexed():
        st.session_state.vector_store = store
        st.session_state.doc_loaded = True
    except Exception:
      pass


all_projects = db_get_projects()
current_proj_idx = (
    all_projects.index(st.session_state.current_project)
    if st.session_state.current_project in all_projects
    else 0
)

selected_project = st.sidebar.selectbox(
    "Select Project",
    all_projects,
    index=current_proj_idx,
    key="project_selector",
    on_change=handle_project_change,
)

with st.sidebar.expander("➕ Create New Project"):
  new_project_name = st.text_input(
      "New Project Name",
      placeholder="e.g., Boardroom_Review_2026",
      key="new_proj_input",
  )
  if st.button("Create Project"):
    if new_project_name and new_project_name not in all_projects:
      db_add_project(new_project_name)
      db_save_settings(
          new_project_name, None, 350, 40, False, "Standard Detailed"
      )
      st.session_state.current_project = new_project_name
      st.session_state.vector_store = None
      st.session_state.doc_loaded = False
      st.session_state.current_kb = None
      st.success(f"Project '{new_project_name}' created!")
      st.rerun()
    elif not new_project_name:
      st.warning("Please enter a valid project name.")
    else:
      st.warning("Project already exists.")

st.sidebar.markdown("---")
st.sidebar.title("📁 Knowledge Base")

active_settings = db_get_settings(st.session_state.current_project)

chunk_size_param = st.sidebar.slider(
    "Chunk Size (words)",
    min_value=100,
    max_value=800,
    value=active_settings["chunk_size"],
    step=50,
)
overlap_param = st.sidebar.slider(
    "Chunk Overlap (words)",
    min_value=10,
    max_value=150,
    value=active_settings["overlap"],
    step=10,
)

reindex_button = st.sidebar.button("🔄 Re-index with New Parameters")

st.sidebar.markdown("---")

clean_active_proj = "".join(
    c if c.isalnum() or c in (".", "_", "-") else "_"
    for c in st.session_state.current_project
).strip("._-")
if not clean_active_proj:
  clean_active_proj = "project"

all_collections = chroma_client.list_collections()
collection_names = [col.name for col in all_collections]
valid_collections = [
    name
    for name in collection_names
    if name.startswith(f"{clean_active_proj}_cs")
    and "_ol" in name
]


def handle_cache_change():
  selected_val = st.session_state.cache_selector
  if selected_val != "-- Choose from cache --":
    try:
      parts = selected_val.rsplit("_cs", 1)
      params = parts[1].split("_ol")
      c_size = int(params[0])
      o_size = int(params[1])

      st.session_state.current_kb = selected_val
      db_save_settings(
          st.session_state.current_project,
          selected_val,
          c_size,
          o_size,
          active_settings["allow_fallback"],
          active_settings["response_style"],
      )

      store = ChromaVectorStore(
          project_name=st.session_state.current_project,
          filename=selected_val,
          chunk_size=c_size,
          overlap=o_size,
      )
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.rerun()
    except Exception as e:
      st.error(f"Error loading cached KB: {e}")
  else:
    st.session_state.current_kb = None
    st.session_state.vector_store = None
    st.session_state.doc_loaded = False
    st.rerun()


if "cache_selector" not in st.session_state:
  st.session_state.cache_selector = (
      active_settings["current_kb"]
      if active_settings["current_kb"] in valid_collections
      else "-- Choose from cache --"
  )

selected_cached_kb = st.sidebar.selectbox(
    f"Cached KBs for '{st.session_state.current_project}':",
    ["-- Choose from cache --"] + valid_collections,
    key="cache_selector",
    on_change=handle_cache_change,
)

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload Files (Multiple PDFs/TXT/MD)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
    key="file_uploader",
)

if uploaded_files:
  if st.sidebar.button("🚀 Process & Index Uploaded Files"):
    file_names = [f.name for f in uploaded_files]
    combined_kb_name = (
        ", ".join(file_names)
        if len(file_names) <= 2
        else f"{file_names[0]} and {len(file_names)-1} others"
    )

    clean_kb_id = "".join(
        c if c.isalnum() or c in (".", "_", "-") else "_"
        for c in combined_kb_name
    ).strip("._-")
    kb_identifier = f"{clean_kb_id}_cs{chunk_size_param}_ol{overlap_param}"

    st.session_state.current_kb = kb_identifier

    db_save_settings(
        st.session_state.current_project,
        kb_identifier,
        chunk_size_param,
        overlap_param,
        active_settings["allow_fallback"],
        active_settings["response_style"],
    )

    try:
      with st.spinner(
          "Extracting, deduplicating, and embedding all uploaded files..."
      ):
        combined_extracted_text = ""
        for uploaded_file in uploaded_files:
          file_name_lower = uploaded_file.name.lower()
          if file_name_lower.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
              page_text = page.extract_text()
              if page_text:
                combined_extracted_text += page_text + "\n"
          else:
            file_bytes = uploaded_file.read()
            combined_extracted_text += (
                file_bytes.decode("utf-8", errors="ignore") + "\n"
            )

        store = ChromaVectorStore(
            project_name=st.session_state.current_project,
            filename=kb_identifier,
            chunk_size=chunk_size_param,
            overlap=overlap_param,
        )
        store.add_document(combined_extracted_text)
        st.session_state.vector_store = store
        st.session_state.doc_loaded = True
        st.sidebar.success(
            "Successfully indexed and combined all uploaded files!"
        )
        st.rerun()
    except Exception as e:
      st.sidebar.error(f"Error processing files: {e}")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🧠 Assistant Behavior")
allow_fallback = st.sidebar.toggle(
    "🌐 Allow General Knowledge Fallback",
    value=active_settings["allow_fallback"],
    help="Allow web search and general knowledge fallback if answers are missing from docs.",
)

response_style = st.sidebar.selectbox(
    "👔 Output Persona",
    [
        "Standard Detailed",
        "Boardroom Executive Briefing",
        "Strategic Action & Risk Plan",
    ],
    index=[
        "Standard Detailed",
        "Boardroom Executive Briefing",
        "Strategic Action & Risk Plan",
    ].index(active_settings["response_style"])
    if active_settings["response_style"]
    in [
        "Standard Detailed",
        "Boardroom Executive Briefing",
        "Strategic Action & Risk Plan",
    ]
    else 0,
)

db_save_settings(
    st.session_state.current_project,
    st.session_state.current_kb,
    chunk_size_param,
    overlap_param,
    allow_fallback,
    response_style,
)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Active Knowledge Base"):
  if st.session_state.current_kb:
    try:
      clean_kb = "".join(
          c if c.isalnum() or c in (".", "_", "-") else "_"
          for c in st.session_state.current_kb
      ).strip("._-")
      col_name = f"{clean_active_proj}_{clean_kb}"
      chroma_client.delete_collection(name=col_name)
    except Exception:
      pass

  db_save_settings(
      st.session_state.current_project,
      None,
      chunk_size_param,
      overlap_param,
      allow_fallback,
      response_style,
  )
  db_clear_chat_history(st.session_state.current_project)
  st.session_state.vector_store = None
  st.session_state.doc_loaded = False
  st.session_state.current_kb = None
  st.rerun()


# ==========================================
# 10. MAIN CHAT INTERFACE & GENERATION PIPELINE
# ==========================================
if not st.session_state.vector_store or not st.session_state.current_kb:
  kb_display = "None Loaded"
else:
  kb_display = st.session_state.current_kb

st.info(
    f"🗂️ Active Project: **{st.session_state.current_project}** | 📂 Knowledge"
    f" Base: **{kb_display}**"
)

chat_container = st.container(height=480)

project_messages = db_get_chat_history(st.session_state.current_project)

with chat_container:
  for message in project_messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

if user_query := st.chat_input(
    "Ask a question about your documents:", key="main_chat_input_box"
):
  if not st.session_state.doc_loaded or not st.session_state.vector_store:
    st.warning(
        "⚠️ Please upload documents or select a cached Knowledge Base in the"
        " sidebar first so the assistant has context to answer from."
    )
  else:
    db_add_chat_message(st.session_state.current_project, "user", user_query)
    project_messages.append({"role": "user", "content": user_query})

    with chat_container:
      with st.chat_message("user"):
        st.markdown(user_query)

      with st.chat_message("assistant"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div class="typing-loader">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
          retrieved_chunks = st.session_state.vector_store.search(
              user_query, top_k=6
          )
          context = "\n\n".join(retrieved_chunks)

          web_context = ""
          if allow_fallback and (not context or len(context.strip()) < 50):
            loading_placeholder.markdown("🌍 Searching live web for info...")
            web_context = perform_web_search(user_query, max_results=3)

          history_str = ""
          for msg in project_messages[-6:-1]:
            content = msg["content"]
            if "I don't know based on the provided documents" in content:
              continue
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {content}\n"

          if response_style == "Boardroom Executive Briefing":
            persona_instructions = """You are a senior executive advisor presenting to a corporate board of directors. 
Your tone must be authoritative, concise, objective, and strategic. 
Structure your response as follows:
1. **Executive Summary:** A brief high-level takeaway.
2. **Key Findings:** Bulleted insights backed directly by the context.
3. **Strategic Implications:** What this means for corporate decision-making."""
          elif response_style == "Strategic Action & Risk Plan":
            persona_instructions = """You are a strategic risk consultant briefing leadership. 
Focus strictly on operational impacts, risks, and actionable next steps derived from the context."""
          else:
            persona_instructions = """You are a helpful AI assistant. Answer the question thoroughly using the provided context."""

          if allow_fallback:
            prompt = f"""{persona_instructions}

PROJECT: {st.session_state.current_project}
KNOWLEDGE BASE: {st.session_state.current_kb}
Fallback Mode: ENABLED (Web Search & General Knowledge Priority).

CRITICAL INSTRUCTIONS FOR ANSWERING:
1. Answer ALL parts of the user's question point-by-point.
2. Check the LOCAL DOCUMENT CONTEXT first. 
3. If the answer is not in the local documents, PRIORITIZE AND USE THE LIVE WEB SEARCH RESULTS provided below.
4. Only if both local documents and web search results lack the information, use your internal general knowledge as a final fallback.

CONVERSATIONAL HISTORY:
{history_str}

LOCAL DOCUMENT CONTEXT:
{context}

LIVE WEB SEARCH RESULTS:
{web_context}

CURRENT QUESTION:
{user_query}

ANSWER:"""
          else:
            prompt = f"""{persona_instructions}

PROJECT: {st.session_state.current_project}
KNOWLEDGE BASE: {st.session_state.current_kb}
Fallback Mode: STRICT (ZERO TOLERANCE FOR EXTERNAL KNOWLEDGE). 

CRITICAL INSTRUCTIONS:
1. Answer ALL parts of the user's question point-by-point using ONLY the LOCAL DOCUMENT CONTEXT.
2. Do NOT use any outside knowledge, assumptions, or training memory.
3. If the exact answer cannot be found in the LOCAL DOCUMENT CONTEXT, your response MUST be precisely and only: "I don't know based on the provided documents."

CONVERSATIONAL HISTORY:
{history_str}

LOCAL DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{user_query}

ANSWER:"""

          response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
          answer = response["response"]

          loading_placeholder.empty()
          st.markdown(answer)
          db_add_chat_message(
              st.session_state.current_project, "assistant", answer
          )
          st.rerun()
        except Exception as e:
          loading_placeholder.empty()
          st.error(
              f"Error generating response from Ollama: {e}. Make sure"
              " Ollama is running (`ollama serve`)."
          )