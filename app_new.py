import os
import chromadb
import numpy as np
import ollama
import streamlit as st
from pypdf import PdfReader

# --- CONFIGURATION ---
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2"

st.set_page_config(
    page_title="Local Ollama Chroma RAG Assistant", page_icon="🤖", layout="wide"
)


# --- CHROMA CLIENT INITIALIZATION ---
@st.cache_resource
def get_chroma_client():
  return chromadb.PersistentClient(path=CHROMA_DIR)


chroma_client = get_chroma_client()


# --- VECTOR STORE CLASS USING CHROMA ---
class ChromaVectorStore:

  def __init__(self, filename, chunk_size=350, overlap=40):
    self.filename = filename
    self.chunk_size = chunk_size
    self.overlap = overlap

    # Clean filename to meet Chroma's strict naming rules (no spaces, only a-zA-Z0-9._-)
    clean_name = "".join(
        c if c.isalnum() or c in (".", "_", "-") else "_"
        for c in self.filename
    )
    clean_name = clean_name.strip("._-")
    if not clean_name or len(clean_name) < 3:
      clean_name = "doc_kb"

    self.collection_name = f"{clean_name}_cs{chunk_size}_ol{overlap}"
    self.collection = chroma_client.get_or_create_collection(
        name=self.collection_name
    )

  def is_indexed(self):
    """Returns True if the collection already contains chunks."""
    return self.collection.count() > 0

  def add_document(self, text):
    words = text.split()
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

    chunk_indices = range(0, len(words), self.chunk_size - self.overlap)
    for idx, i in enumerate(chunk_indices):
      chunk = " ".join(words[i : i + self.chunk_size])
      if len(chunk.strip()) > 30:
        try:
          response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=chunk)
          vector = response["embedding"]

          chunks.append(chunk)
          embeddings.append(vector)
          ids.append(f"chunk_{idx}")
          metadatas.append(
              {"source": self.filename, "chunk_index": str(idx)}
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
st.title("🤖 Local Ollama Chroma RAG Assistant")
st.markdown(
    f"Ask questions below. Powered by Chroma DB and Ollama (`{GENERATION_MODEL}`"
    f")."
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

# Cached Collections Selector from Chroma DB
# Cached Collections Selector from Chroma DB (Showing full collection names including parameters)
all_collections = chroma_client.list_collections()
collection_names = [col.name for col in all_collections]

# Filter to only show collections that match our RAG naming pattern
valid_collections = [name for name in collection_names if "_cs" in name and "_ol" in name]

selected_cached_kb = st.sidebar.selectbox(
    "Or Select Existing Cached KB (by parameters):", 
    ["-- Choose from cache --"] + valid_collections
)

if selected_cached_kb != "-- Choose from cache --":
  # Parse back the filename, chunk size, and overlap from the collection name format: filename_csX_olY
  try:
    parts = selected_cached_kb.rsplit("_cs", 1)
    base_file = parts[0]
    params = parts[1].split("_ol")
    c_size = int(params[0])
    o_size = int(params[1])
    
    if st.session_state.current_kb != selected_cached_kb:
      store = ChromaVectorStore(
          filename=base_file,
          chunk_size=c_size,
          overlap=o_size,
      )
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.session_state.current_kb = selected_cached_kb
      st.sidebar.success(f"Loaded '{selected_cached_kb}' from Chroma DB!")
  except Exception as e:
    st.sidebar.error(f"Error loading collection: {e}")

st.sidebar.markdown("---")

# File Uploader (Accepts TXT, MD, PDF)
uploaded_file = st.sidebar.file_uploader(
    "Or Upload File", type=["txt", "md", "pdf"]
)

if uploaded_file is not None:
  file_base_name = uploaded_file.name
  if st.session_state.current_kb != file_base_name:
    st.session_state.current_kb = file_base_name
    try:
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

# Handle Document Loading / Re-indexing Logic with Chroma
if st.session_state.get("current_kb"):
  store = ChromaVectorStore(
      filename=st.session_state.current_kb,
      chunk_size=chunk_size_param,
      overlap=overlap_param,
  )

  if store.is_indexed() and not reindex_button:
    st.session_state.vector_store = store
    st.session_state.doc_loaded = True
    st.sidebar.success(
        "⚡ Vectors loaded instantly from Chroma DB collection!"
    )
  else:
    if st.session_state.get("raw_document_text"):
      st.sidebar.info("Computing embeddings & saving to Chroma DB...")
      store.add_document(st.session_state.raw_document_text)
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True
      st.sidebar.success("Indexed & saved to Chroma DB successfully!")
    elif store.is_indexed():
      st.session_state.vector_store = store
      st.session_state.doc_loaded = True

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Active Knowledge Base"):
  if st.session_state.current_kb:
    try:
      clean_name = "".join(
          c if c.isalnum() or c in (".", "_", "-") else "_"
          for c in st.session_state.current_kb
      )
      clean_name = clean_name.strip("._-")
      col_name = f"{clean_name}_cs{chunk_size_param}_ol{overlap_param}"
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
        with st.spinner("Searching Chroma DB & generating response..."):
          # 1. High-speed Chroma vector search
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