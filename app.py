import streamlit as st
import os
import tempfile
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Easy Multi-PDF Chatbot", page_icon="📚", layout="wide")
st.title("📚 Easy Multi-PDF Chatbot (Powered by LlamaIndex & Groq)")
st.write("A simplified, robust alternative to LangChain for chatting with your documents.")

# --- SIDEBAR FOR CREDENTIALS & UPLOADS ---
with st.sidebar:
    st.header("1. API Configuration")
    groq_api_key = st.text_input(
        "Enter Groq API Key", 
        value=os.environ.get("GROQ_API_KEY", ""), 
        type="password",
        placeholder="gsk_..."
    )
    
    st.markdown("---")
    st.header("2. Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    process_button = st.button("Process Documents", type="primary")

# --- APP STATE & INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# --- PROCESS DOCUMENTS WITH LLAMAINDEX ---
if process_button:
    if not groq_api_key:
        st.error("Please add your Groq API Key to continue.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        with st.spinner("Processing documents... LlamaIndex is handling the chunking and embedding automatically."):
            try:
                # 1. Configure global LlamaIndex settings
                Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.2)
                Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # 2. Save uploaded files to a temporary directory for SimpleDirectoryReader to parse
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                    
                    # LlamaIndex extracts, chunks, and prepares files automatically
                    documents = SimpleDirectoryReader(temp_dir).load_data()
                
                # 3. Build the in-memory vector index (uses a built-in vector store similar to FAISS)
                index = VectorStoreIndex.from_documents(documents)
                
                # 4. Create a chat engine with built-in conversational buffer memory
                st.session_state.chat_engine = index.as_chat_engine(
                    chat_mode="condense_plus_context",
                    verbose=False
                )
                st.success(f"Successfully indexed {len(uploaded_files)} document(s)! Ask your questions below.")
                
            except Exception as e:
                st.error(f"An error occurred during indexing: {str(e)}")

# --- CONVERSATIONAL CHAT INTERFACE ---
# Render historical messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input queries
if user_query := st.chat_input("Ask a question about your documents..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    if not groq_api_key:
        st.error("Missing Groq API credentials. Add your API key in the sidebar configuration.")
    elif st.session_state.chat_engine is None:
        st.error("No knowledge database found. Please upload your PDFs and click 'Process Documents' first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    # The chat engine natively keeps track of conversation context and retrieves relevant facts
                    response = st.session_state.chat_engine.chat(user_query)
                    st.markdown(response.response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response.response})
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
