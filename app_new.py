import hashlib
import json
import os
import re
import chromadb
import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

# --- CONFIGURATION ---
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
PROJECTS_FILE = os.path.join(CHROMA_DIR, "projects.json")

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

st.set_page_config(
    page_title="Local Ollama Chroma RAG Assistant", page_icon="🤖", layout="wide"
)

# --- CUSTOM CSS FOR GEMINI/CLAUDE STYLE LOADER ANIMATION ---
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


# --- PERSISTENT PROJECT UTILITIES ---
def load_saved_projects():
  if os.path.exists(PROJECTS_FILE):
    try:
      with open(PROJECTS_FILE, "r") as f:
        data = json.load(f)
        if data:
          return data
    except Exception:
      pass
  return ["Default Project"]


def save_projects_to_disk(projects_list):
  try:
    with open(PROJECTS_FILE, "w") as f:
      json.dump(projects_list, f)
  except Exception as e:
    st.warning(f"Could not save projects: {e}")


# --- TEXT CLEANING UTILITY ---
def clean_text(text: str) -> str:
  """Cleans raw extracted document text by removing extra whitespaces

  and normalizing formatting artifacts.
  """
  if not text:
    return ""
  text = re.sub(r"\n\s*\n", "\n\n", text)
  text = re.sub(r"[ \t]+", " ", text)
  return text.strip()


# --- CHROMA CLIENT INITIALIZATION ---
@st.cache_resource
def get_chroma_client():
  return chromadb.PersistentClient(path=CHROMA_DIR)


chroma_client = get_chroma_client()


# --- VECTOR STORE CLASS USING CHROMA & PROJECTS ---
class ChromaVectorStore:

  def __init__(self, project_name, filename, chunk_size=350, overlap=40):
    self.project_name = project_name
    self.filename = filename
    self.chunk_size = chunk_size
    self.overlap = overlap

    # Namespace collection strictly by project hash and file hash
    proj_hash = hashlib.md5(self.project_name.encode("utf-8")).hexdigest()[:6]
    file_hash = hashlib.md5(self.filename.encode("utf-8")).hexdigest()[:8]

    self.collection_name = f"p_{proj_hash}_kb_{file_hash}_cs{chunk_size}_ol{overlap}"
    self.collection = chroma_client.get_or_create_collection(
        name=self.collection_name
    )

  def is_indexed(self):
    """Returns True if the collection already contains chunks."""
    return self.collection.count() > 0

  def add_document(self, text):
    cleaned_text = clean_text(text)
    words = cleaned_text.split()
    if not words:
      return

    # Clear existing items if re-indexing
    existing_ids = self.collection.get()["ids"]
    if existing_ids:
      self.collection.delete(ids=existing_ids)

    chunks = []
    embeddings = []
    ids = []
    metadatas = []

    seen_chunk_hashes = set()
    chunk_indices = range(0, len(words), self.chunk_size - self.overlap)
    
    for idx, i in enumerate(chunk_indices):
      chunk = " ".join(words[i : i + self.chunk_size])
      
      # Deduplicate chunks using MD5 hashing for chunks > 30 characters
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
          st.warning(f"Error embedding chunk {idx}: {e}")

    if chunks:
      self.collection.add(
          documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas
      )

  def search(self, query, top_k=5):
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


# --- INITIALIZE SESSION STATE WITH PERSISTENCE ---
if "projects" not in st.session_state:
  st.session_state.projects = load_saved_projects()
if "current_project" not in st.session_state:
  st.session_state.current_project = st.session_state.projects[0]
if "vector_store" not in st.session_state:
  st.session_state.vector_store = None
if "doc_loaded" not in st.session_state:
  st.session_state.doc_loaded = False
if "current_kb" not in st.session_state:
  st.session_state.current_kb = None
if "raw_document_text" not in st.session_state:
  st.session_state.raw_document_text = None
if "messages" not in st.session_state:
  st.session_state.messages = []

# --- UI HEADER ---
st.title("🤖 Local Ollama Chroma RAG Assistant")
st.markdown(
    f"Ask questions below. Powered by Chroma DB and Ollama (`{GENERATION_MODEL}`"
    f")."
)

# --- SIDEBAR: PROJECT & KNOWLEDGE BASE MANAGEMENT ---
st.sidebar.title("🗂️ Project Workspace")


def handle_project_change():
  st.session_state.current_project = st.session_state.project_selector
  st.session_state.vector_store = None
  st.session_state.doc_loaded = False
  st.session_state.current_kb = None
  st.session_state.raw_document_text = None
  st.session_state.messages = []


current_proj_idx = 0
if st.session_state.current_project in st.session_state.projects:
  current_proj_idx = st.session_state.projects.index(
      st.session_state.current_project
  )

selected_project = st.sidebar.selectbox(
    "Select Project",
    st.session_state.projects,
    index=current_proj_idx,
    key="project_selector",
    on_change=handle_project_change,
)

# Explicit Create Project Section
with st.sidebar.expander("➕ Create New Project"):
  new_project_name = st.text_input(
      "New Project Name",
      placeholder="e.g., Financial_Analysis_2026",
      key="new_proj_input",
  )
  if st.button("Create Project"):
    if new_project_name and new_project_name not in st.session_state.projects:
      st.session_state.projects.append(new_project_name)
      save_projects_to_disk(st.session_state.projects)
      st.session_state.current_project = new_project_name
      st.session_state.vector_store = None
      st.session_state.doc_loaded = False
      st.session_state.current_kb = None
      st.session_state.raw_document_text = None
      st.session_state.messages = []
      st.success(f"Project '{new_project_name}' created!")
      st.rerun()
    elif not new_project_name:
      st.warning("Please enter a valid project name.")
    else:
      st.warning("Project already exists.")

st.sidebar.markdown("---")
st.sidebar.title("📁 Knowledge Base")

# Dynamic Chunking Sliders
st.sidebar.markdown("### ⚙️ Chunking Parameters")
chunk_size_param = st.sidebar.slider(
    "Chunk Size (words)", min_value=100, max_value=800, value=350, step=50
)
overlap_param = st.sidebar.slider(
    "Chunk Overlap (words)", min_value=10, max_value=150, value=40, step=10
)

# Re-index Control Button
reindex_button = st.sidebar.button("🔄 Re-index with New Parameters")

st.sidebar.markdown("---")

# Fallback Toggle Switch
st.sidebar.markdown("### 🧠 Assistant Behavior")
allow_fallback = st.sidebar.toggle(
    "🌐 Allow General Knowledge Fallback",
    value=False,
    help=(
        "If enabled, the model can use its general knowledge when an answer is"
        " not found in the documents."
    ),
)

st.sidebar.markdown("---")

# Filter Cached Collections strictly by Active Project Hash
active_proj_hash = hashlib.md5(
    st.session_state.current_project.encode("utf-8")
).hexdigest()[:6]
all_collections = chroma_client.list_collections()
collection_names = [col.name for col in all_collections]

valid_collections = [
    name
    for name in collection_names
    if name.startswith(f"p_{active_proj_hash}_")
    and "_cs" in name
    and "_ol" in name
]

selected_cached_kb = st.sidebar.selectbox(
    f"Cached KBs for '{st.session_state.current_project}':",
    ["-- Choose from cache --"] + valid_collections,
    key="cache_selector",
)

if selected_cached_kb != "-- Choose from cache --":
  try:
    parts = selected_cached_kb.split("_kb_")
    rest = parts[1]

    file_and_params = rest.rsplit("_cs", 1)
    base_prefix = file_and_params[0]
    params = file_and_params[1].split("_ol")
    c_size = int(params[0])
    o_size = int(params[1])

    if st.session_state.current_kb != selected_cached_kb:
      store = ChromaVectorStore(
          project_name=st.session_state.current_project,
          filename=base_prefix,
          chunk_size=c_size,
          overlap=o_size,
      )
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.session_state.current_kb = selected_cached_kb
      st.sidebar.success("Loaded collection from cache!")
  except Exception as e:
    st.sidebar.error(f"Error loading collection: {e}")

st.sidebar.markdown("---")

# Multi-File Uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload Files (Multiple PDFs/TXT/MD)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True,
    key="file_uploader",
)

if uploaded_files:
  file_names = [f.name for f in uploaded_files]
  combined_kb_name = (
      ", ".join(file_names)
      if len(file_names) <= 2
      else f"{file_names[0]} and {len(file_names)-1} others"
  )

  if st.session_state.current_kb != combined_kb_name:
    st.session_state.current_kb = combined_kb_name
    try:
      combined_extracted_text = ""
      for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".pdf"):
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
      st.session_state.raw_document_text = combined_extracted_text
    except Exception as e:
      st.sidebar.error(f"Error reading files: {e}")

# Handle Document Loading / Re-indexing Logic with Chroma
if st.session_state.get("current_kb"):
  store = ChromaVectorStore(
      project_name=st.session_state.current_project,
      filename=st.session_state.current_kb,
      chunk_size=chunk_size_param,
      overlap=overlap_param,
  )

  if store.is_indexed() and not reindex_button:
    st.session_state.vector_store = store
    st.session_state.doc_loaded = True
    st.sidebar.success("⚡ Vectors loaded instantly from Chroma DB!")
  else:
    if st.session_state.get("raw_document_text"):
      st.sidebar.info("Cleaning, deduplicating, and embedding documents...")
      store.add_document(st.session_state.raw_document_text)
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.sidebar.success("Saved successfully to Chroma DB!")
    elif store.is_indexed():
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Active Knowledge Base"):
  if st.session_state.current_kb:
    try:
      proj_hash = hashlib.md5(
          st.session_state.current_project.encode("utf-8")
      ).hexdigest()[:6]
      file_hash = hashlib.md5(
          st.session_state.current_kb.encode("utf-8")
      ).hexdigest()[:8]
      col_name = (
          f"p_{proj_hash}_kb_{file_hash}_cs{chunk_size_param}_ol{overlap_param}"
      )
      chroma_client.delete_collection(name=col_name)
    except Exception:
      pass

  st.session_state.vector_store = None
  st.session_state.doc_loaded = False
  st.session_state.current_kb = None
  st.session_state.raw_document_text = None
  st.session_state.messages = []
  st.rerun()

# --- MAIN CHAT INTERFACE ---
if st.session_state.doc_loaded and st.session_state.current_kb:
  st.info(
      f"🗂️ Active Project: **{st.session_state.current_project}** | 📂 Knowledge"
      f" Base: **{st.session_state.current_kb}**"
  )

  chat_container = st.container(height=480)

  with chat_container:
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  if user_query := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": user_query})

    with chat_container:
      with st.chat_message("user"):
        st.markdown(user_query)

      with st.chat_message("assistant"):
        # Placeholder for the Gemini/Claude style bouncing dots animation
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
          # 1. Search Chroma DB
          retrieved_chunks = st.session_state.vector_store.search(
              user_query, top_k=4
          )
          context = "\n\n".join(retrieved_chunks)

          history_str = ""
          for msg in st.session_state.messages[-6:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

          if allow_fallback:
            prompt = f"""You are a helpful AI assistant. Answer the question using the provided document context and conversational history. 
If the information is present in the document context, rely on it. 
If the information is NOT in the document context, you are allowed to use your own general knowledge to answer, but please note to the user that the information is coming from general knowledge rather than the uploaded document.

CONVERSATIONAL HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{user_query}

ANSWER:"""
          else:
            prompt = f"""You are a strict, helpful AI assistant. Answer the question using the provided document context and conversational history. If the answer cannot be found within the context, state clearly that the document does not contain the information.

CONVERSATIONAL HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

CURRENT QUESTION:
{user_query}

ANSWER:"""

          # 2. Generate response from Ollama
          response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
          answer = response["response"]

          # Clear the animation placeholder and display the final markdown answer
          loading_placeholder.empty()
          st.markdown(answer)
          st.session_state.messages.append(
              {"role": "assistant", "content": answer}
          )
        except Exception as e:
          loading_placeholder.empty()
          st.error(
              f"Error generating response from Ollama: {e}. Make sure"
              " Ollama is running (`ollama serve`)."
          )
else:
  st.sidebar.warning(
      "👈 Please select or create a project and upload one or more documents in"
      " the sidebar to begin."
  )