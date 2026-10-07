import os
import pickle
import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader  # Added for reading PDF files

# --- CONFIGURATION ---
CACHE_DIR = "vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

st.set_page_config(
    page_title="Local Ollama Strict RAG Assistant", page_icon="🤖", layout="wide"
)


# --- VECTOR STORE CLASS ---
class OllamaVectorStore:

  def __init__(self, filename, chunk_size=350, overlap=40):
    self.filename = filename
    self.chunk_size = chunk_size
    self.overlap = overlap
    self.chunks = []
    self.embeddings = []

  def get_cache_path(self):
    clean_name = "".join(
        c if c.isalnum() or c in (" ", "_", "-") else "_"
        for c in self.filename
    )
    return os.path.join(
        CACHE_DIR, f"{clean_name}_cs{self.chunk_size}_ol{self.overlap}.pkl"
    )

  def save_cache(self):
    cache_path = self.get_cache_path()
    try:
      with open(cache_path, "wb") as f:
        pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
    except Exception as e:
      st.error(f"Error saving cache: {e}")

  def load_cache(self):
    cache_path = self.get_cache_path()
    if os.path.exists(cache_path):
      try:
        with open(cache_path, "rb") as f:
          data = pickle.load(f)
          self.chunks = data.get("chunks", [])
          self.embeddings = data.get("embeddings", [])
          return True
      except Exception as e:
        st.error(f"Error loading cache: {e}")
    return False

  def add_document(self, text):
    words = text.split()
    self.chunks = []
    self.embeddings = []

    if not words:
      return

    chunk_indices = range(0, len(words), self.chunk_size - self.overlap)
    for i in chunk_indices:
      chunk = " ".join(words[i : i + self.chunk_size])
      if len(chunk.strip()) > 30:
        try:
          response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
          vector = response["embedding"]
          self.chunks.append(chunk)
          self.embeddings.append(vector)
        except Exception as e:
          st.warning(f"Error embedding chunk: {e}")

    self.save_cache()

  def search(self, query, top_k=5):
    if not self.chunks:
      return []

    try:
      res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
      query_vector = np.array(res["embedding"])
    except Exception as e:
      st.error(f"Error generating query embedding: {e}")
      return []

    similarities = []
    for emb in self.embeddings:
      emb_vector = np.array(emb)
      norm_product = np.linalg.norm(query_vector) * np.linalg.norm(emb_vector)
      if norm_product == 0:
        sim = 0.0
      else:
        sim = np.dot(query_vector, emb_vector) / norm_product
      similarities.append(sim)

    if not similarities:
      return []

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [self.chunks[i] for i in top_indices if i < len(self.chunks)]


# --- INITIALIZE SESSION STATE ---
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
st.title("🤖 Local Ollama Strict RAG Assistant")
st.markdown(
    f"Ask questions below. Completely offline, private, and free using Ollama"
    f" (`{GENERATION_MODEL}`)."
)

# --- SIDEBAR: KNOWLEDGE BASE MANAGEMENT ---
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
        " not found in the document."
    ),
)

st.sidebar.markdown("---")

# Cached Files Selector
cached_files = [f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")]
selected_cached_kb = st.sidebar.selectbox(
    "Or Select Existing Cached KB:", ["-- Choose from cache --"] + cached_files
)

if (
    selected_cached_kb != "-- Choose from cache --"
    and st.session_state.current_kb != selected_cached_kb
):
  base_kb_name = selected_cached_kb
  if "_cs" in selected_cached_kb and "_ol" in selected_cached_kb:
    base_kb_name = selected_cached_kb.split("_cs")[0]

  store = OllamaVectorStore(
      filename=base_kb_name,
      chunk_size=chunk_size_param,
      overlap=overlap_param,
  )
  if store.load_cache():
    st.session_state.vector_store = store
    st.session_state.doc_loaded = True
    st.session_state.current_kb = base_kb_name
    st.sidebar.success(f"Loaded '{base_kb_name}' from cache!")
  else:
    st.sidebar.error("Could not load cache for these specific parameters.")

st.sidebar.markdown("---")

# File Uploader updated to accept PDF along with TXT and MD
uploaded_file = st.sidebar.file_uploader(
    "Or Upload File", type=["txt", "md", "pdf"]
)

if uploaded_file is not None:
  file_base_name = uploaded_file.name
  if st.session_state.current_kb != file_base_name:
    st.session_state.current_kb = file_base_name
    try:
      # Extract text depending on whether it's a PDF or text/md file
      if file_base_name.lower().endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        extracted_text = ""
        for page in pdf_reader.pages:
          page_text = page.extract_text()
          if page_text:
            extracted_text += page_text + "\n"
        st.session_state.raw_document_text = extracted_text
      else:
        file_bytes = uploaded_file.read()
        st.session_state.raw_document_text = file_bytes.decode(
            "utf-8", errors="ignore"
        )
    except Exception as e:
      st.sidebar.error(f"Error reading file: {e}")

# Handle Document Loading / Re-indexing Logic
if st.session_state.get("current_kb") and (
    reindex_button or not st.session_state.doc_loaded
):
  if not st.session_state.get("raw_document_text"):
    store = OllamaVectorStore(
        filename=st.session_state.current_kb,
        chunk_size=chunk_size_param,
        overlap=overlap_param,
    )
    if store.load_cache():
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.sidebar.success("⚡ Loaded embeddings instantly from cache!")
    else:
      st.sidebar.warning(
          "Please re-upload your file to compute embeddings with new"
          " parameters."
      )
  else:
    store = OllamaVectorStore(
        filename=st.session_state.current_kb,
        chunk_size=chunk_size_param,
        overlap=overlap_param,
    )

    if not reindex_button and store.load_cache():
      st.sidebar.success("⚡ Loaded embeddings instantly from local cache!")
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
    else:
      st.sidebar.info("Computing embeddings (fresh indexing)...")
      store.add_document(st.session_state.raw_document_text)
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.sidebar.success("Indexed & cached successfully!")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Active Knowledge Base"):
  st.session_state.vector_store = None
  st.session_state.doc_loaded = False
  st.session_state.current_kb = None
  st.session_state.raw_document_text = None
  st.session_state.messages = []
  st.rerun()

# --- MAIN CHAT INTERFACE ---
if st.session_state.doc_loaded and st.session_state.current_kb:
  st.info(f"📂 Active Knowledge Base: **{st.session_state.current_kb}**")

  # Scrollable chat history container
  chat_container = st.container(height=480)

  with chat_container:
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  # Chat input box
  if user_query := st.chat_input("Ask a question about your document:"):
    st.session_state.messages.append({"role": "user", "content": user_query})

    with chat_container:
      with st.chat_message("user"):
        st.markdown(user_query)

      with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & generating response..."):
          # 1. Vector similarity search
          retrieved_chunks = st.session_state.vector_store.search(
              user_query, top_k=4
          )
          context = "\n\n".join(retrieved_chunks)

          # 2. Extract recent conversational memory
          history_str = ""
          for msg in st.session_state.messages[-6:-1]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

          # 3. Prompt customization based on Fallback Toggle
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

          try:
            response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
            answer = response["response"]
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
          except Exception as e:
            st.error(
                f"Error generating response from Ollama: {e}. Make sure"
                " Ollama is running (`ollama serve`)."
            )
else:
  st.warning(
      "👈 Please select an existing cached knowledge base or upload a text/.md/.pdf"
      " file in the sidebar to begin."
  )