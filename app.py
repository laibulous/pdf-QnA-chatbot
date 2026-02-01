# --- 1. THE SQLITE FIX (CRITICAL FOR CLOUD DEPLOYMENT) ---
# This must be at the very top before any other imports!
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. STANDARD IMPORTS ---
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with PDF (Powered by Gemini)")

# --- 4. API KEY SETUP ---
# We look for the key in Streamlit Secrets first, then environment variable
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    # Fallback for local testing if secrets.toml isn't set up
    api_key = st.sidebar.text_input("Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

if "GOOGLE_API_KEY" not in os.environ:
    st.warning("Please enter your Google API Key in the sidebar to continue.")
    st.stop()

# --- 5. FUNCTIONS (Cached for Performance) ---

@st.cache_resource
def process_pdf(file):
    """
    Saves the uploaded file, loads it, splits it, and returns the vector store.
    """
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    # Load and Split
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create Embeddings & Vector Store
    # using Local Embeddings (CPU friendly)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Cleanup temp file
    os.remove(tmp_path)
    return vectorstore

@st.cache_resource
def get_rag_chain(_vectorstore):
    """
    Creates the RAG chain using the vector store.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        temperature=0
    )

    retriever = _vectorstore.as_retriever()

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- 6. MAIN UI ---

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF... (This may take a moment)"):
        try:
            vectorstore = process_pdf(uploaded_file)
            st.success("PDF Processed!")
            
            # Initialize Chain
            rag_chain = get_rag_chain(vectorstore)

            # Chat Interface
            query = st.text_input("Ask a question about your document:")
            
            if query:
                with st.spinner("Thinking..."):
                    response = rag_chain.invoke({"input": query})
                    st.write("### Answer")
                    st.write(response["answer"])
                    
                    # Optional: Show context for debugging
                    with st.expander("See retrieved context"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(doc.page_content)
                            
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
