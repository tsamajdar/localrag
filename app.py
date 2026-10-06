import os
import pickle
import numpy as np
import ollama
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Local Ollama Strict RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"
CACHE_DIR = "vector_cache"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


# --- LIGHTWEIGHT LOCAL VECTOR STORE WITH CACHING ---
class OllamaVectorStore:

  def __init__(self, filename=""):
    self.full_text = ""
    self.chunks = []
    self.embeddings = []
    self.cache_file = (
        os.path.join(CACHE_DIR, f"{filename}.pkl") if filename else None
    )

  def load_cache(self):
    """Loads pre-computed embeddings from disk if available."""
    if self.cache_file and os.path.exists(self.cache_file):
      try:
        with open(self.cache_file, "rb") as f:
          data = pickle.load(f)
          self.full_text = data.get("full_text", "")
          self.chunks = data.get("chunks", [])
          self.embeddings = data.get("embeddings", [])
        return True
      except Exception:
        return False
    return False

  def save_cache(self):
    """Saves computed embeddings to disk for future instant loading."""
    if self.cache_file:
      try:
        data = {
            "full_text": self.full_text,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
        }
        with open(self.cache_file, "wb") as f:
          pickle.dump(data, f)
      except Exception as e:
        st.warning(f"Could not save cache: {e}")

  def add_document(self, text, chunk_size=400, overlap=40):
    self.full_text = text
    words = text.split()

    progress_bar = st.progress(0)
    total_words = len(words)
    chunk_indices = list(range(0, total_words, chunk_size - overlap))
    total_chunks = len(chunk_indices)

    for idx, i in enumerate(chunk_indices):
      chunk = " ".join(words[i : i + chunk_size])
      if len(chunk.strip()) > 30:
        try:
          response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
          vector = response["embedding"]
          self.chunks.append(chunk)
          self.embeddings.append(vector)
        except Exception as e:
          st.warning(f"Error embedding chunk: {e}")

      if total_chunks > 0:
        progress_bar.progress(min((idx + 1) / total_chunks, 1.0))

    progress_bar.empty()
    self.save_cache()

  def search(self, query, top_k=5):
    if not self.chunks:
      return []

    res = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    query_vector = np.array(res["embedding"])

    similarities = []
    for emb in self.embeddings:
      emb_vector = np.array(emb)
      norm_product = np.linalg.norm(query_vector) * np.linalg.norm(emb_vector)
      if norm_product == 0:
        sim = 0.0
      else:
        sim = np.dot(query_vector, emb_vector) / norm_product
      similarities.append(sim)

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [self.chunks[i] for i in top_indices]


# Initialize session state
if "vector_store" not in st.session_state:
  st.session_state.vector_store = None
if "doc_loaded" not in st.session_state:
  st.session_state.doc_loaded = False
if "current_kb" not in st.session_state:
  st.session_state.current_kb = None

# --- SIDEBAR: KNOWLEDGE BASE MANAGEMENT ---
st.sidebar.title("📁 Knowledge Base")

# Scan vector_cache folder for existing cached files
cached_files = [
    f[:-4] for f in os.listdir(CACHE_DIR) if f.endswith(".pkl")
]

selected_cached_kb = st.sidebar.selectbox(
    "Or Select Existing Cached KB:", ["-- Choose from cache --"] + cached_files
)

if (
    selected_cached_kb != "-- Choose from cache --"
    and st.session_state.current_kb != selected_cached_kb
):
  store = OllamaVectorStore(filename=selected_cached_kb)
  if store.load_cache():
    st.session_state.vector_store = store
    st.session_state.doc_loaded = True
    st.session_state.current_kb = selected_cached_kb
    st.sidebar.success(f"Loaded '{selected_cached_kb}' from cache!")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Or Upload New Text/MD File", type=["txt", "md"]
)

if uploaded_file is not None:
  file_base_name = uploaded_file.name
  if st.session_state.current_kb != file_base_name:
    store = OllamaVectorStore(filename=file_base_name)

    if store.load_cache():
      st.sidebar.success("⚡ Loaded embeddings instantly from local cache!")
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.session_state.current_kb = file_base_name
    else:
      document_text = ""
      try:
        file_bytes = uploaded_file.read()
        document_text = file_bytes.decode("utf-8", errors="ignore")
      except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

      if len(document_text.strip()) == 0:
        st.sidebar.error("The uploaded file is empty!")
      else:
        st.sidebar.info("Computing embeddings (first time only)...")
        store.add_document(document_text)
        st.session_state.vector_store = store
        st.session_state.doc_loaded = True
        st.session_state.current_kb = file_base_name
        st.sidebar.success(f"Indexed & cached '{file_base_name}' successfully!")

if st.sidebar.button("🗑️ Clear Active Knowledge Base"):
  st.session_state.vector_store = None
  st.session_state.doc_loaded = False
  st.session_state.current_kb = None
  st.rerun()

# --- MAIN INTERFACE (Always Available) ---
st.title("🤖 Local Ollama Strict RAG Assistant")
st.markdown(
    "Ask questions below. Completely offline, private, and free using Ollama"
    f" (`{GENERATION_MODEL}`)."
)

if not st.session_state.doc_loaded or st.session_state.vector_store is None:
  st.warning(
      "⚠️ No Knowledge Base selected. Please choose an existing cached database"
      " from the sidebar dropdown or upload a new file."
  )
else:
  st.info(f"📂 Active Knowledge Base: **{st.session_state.current_kb}**")

  user_query = st.text_input(
      "Ask a question about your document:",
      placeholder="e.g., Write in a simple way about Hugging Face",
  )

  if user_query:
    with st.spinner("Searching local vector database and generating response..."):
      query_lower = user_query.lower()
      if any(
          kw in query_lower
          for kw in [
              "simple",
              "overview",
              "what is",
              "explain",
              "summary",
              "about",
          ]
      ):
        all_context = st.session_state.vector_store.full_text[:5000]
        retrieved_chunks = ["(Using full document introductory context)"]
      else:
        retrieved_chunks = st.session_state.vector_store.search(
            user_query, top_k=5
        )
        all_context = "\n\n---\n\n".join(retrieved_chunks)

      prompt = f"""You are a helpful assistant. Explain the answer to the user query clearly and simply, using only the provided context.

CONTEXT:
{all_context}

QUERY:
{user_query}

EXPLANATION:"""

      try:
        response = ollama.generate(model=GENERATION_MODEL, prompt=prompt)
        answer_text = response["response"]

        st.markdown("### Answer")
        st.write(answer_text)

        with st.expander("🔍 View Retrieved Document Chunks (Context Used)"):
          for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.text(chunk)
      except Exception as e:
        st.error(
            f"Error communicating with Ollama. Make sure Ollama is running: {e}"
        )