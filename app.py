import streamlit as st
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF Chatbot using Groq & FAISS")
st.write("Upload your PDF documents, process them into a semantic database, and chat natively with their content.")

# --- SIDEBAR FOR CREDENTIALS & UPLOADS ---
with st.sidebar:
    st.header("1. API Configuration")
    # Streamlit Cloud will securely fetch this from App Secrets if available, otherwise prompt the user
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

# --- CORE RAG OPERATIONS ---
def get_pdf_text(pdf_docs):
    """Extract raw text from uploaded PDF objects."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_text_chunks(text):
    """Split clean text into semantic overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Embed text chunks and build a temporary FAISS vector index."""
    # Using a fast, highly accurate local sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversational_rag_chain(vectorstore, api_key):
    """Construct an end-to-end conversation-aware RAG pipeline."""
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.1-8b-instant", 
        temperature=0.2
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 1. Contextualize the conversation history into standalone search queries
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # 2. Design the Q&A generation block using strictly retrieved files context
    system_prompt = (
        "You are an expert AI document assistant. Answer the user's question using the provided context below. "
        "If you do not know the answer or if it's not found in the context, explicitly say that you cannot find it "
        "in the uploaded documents. Do not make up answers.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- APP LIFECYCLE MANAGEMENT ---
# Initialize persistent chat history and vector index storage across user interactions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Trigger document indexing when the process button is clicked
if process_button:
    if not groq_api_key:
        st.error("Please add your Groq API Key in the sidebar configuration to continue.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        with st.spinner("Processing documents (Extracting, chunking, and indexing into FAISS)..."):
            raw_text = get_pdf_text(uploaded_files)
            if not raw_text.strip():
                st.error("Failed to extract readable text from the uploaded PDFs. Please ensure they aren't scanned images.")
            else:
                text_chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = get_vectorstore(text_chunks)
                st.success(f"Successfully processed {len(uploaded_files)} document(s)! Database is ready.")

# --- CONVERSATIONAL CHAT INTERFACE ---
# Render past messages in the conversational UI window
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

# Accept real-time user inputs
if user_query := st.chat_input("Ask a question about your documents..."):
    # Append human input visually
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    # Fallback checks before making an LLM API call
    if not groq_api_key:
        st.error("Missing Groq API credentials. Add your API key in the sidebar configuration.")
    elif st.session_state.vector_store is None:
        st.error("No knowledge database found. Please upload your PDFs and click 'Process Documents' first.")
    else:
        with st.chat_message("AI"):
            with st.spinner("Searching document database and generating answer..."):
                try:
                    # Construct and invoke the conversational retrieval chain
                    rag_chain = get_conversational_rag_chain(st.session_state.vector_store, groq_api_key)
                    response = rag_chain.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Update local state history to maintain memory context
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                except Exception as e:
                    st.error(f"An error occurred while calling the Groq LLM: {str(e)}")

