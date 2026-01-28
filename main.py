import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. SETUP API KEY ---
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

# --- 2. LOAD PDF ---
print("Loading PDF...")
# Ensure "your_document.pdf" exists in the folder!
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# --- 3. SPLIT TEXT ---
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# --- 4. EMBEDDINGS (Local & Free) ---
print("Creating embeddings...")
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
retriever = vectorstore.as_retriever()

# --- 5. INITIALIZE GEMINI ---
print("Initializing Gemini...")
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0
)

# --- 6. CREATE THE CHAIN (The Fix) ---
# You MUST define this prompt for create_stuff_documents_chain to work
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the "
    "answer concise and readable."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# This chains the LLM and the Prompt together
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# This connects the Retriever (Database) to the Q&A Chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 7. ASK QUESTION ---
query = input("Ask a questionn\n")
print(f"\nThinking about: {query}...\n")

# Note: We use "input" here, not "question"
response = rag_chain.invoke({"input": query})

print("--- RESPONSE ---")
print(response["answer"])

#AVAILABLE MODELS FOR FREE TIER GEMINI API:
# -----------------
# - models/gemini-2.5-flash
# - models/gemini-2.5-pro
# - models/gemini-2.0-flash
# - models/gemini-2.0-flash-001
# - models/gemini-2.0-flash-exp-image-generation
# - models/gemini-2.0-flash-lite-001
# - models/gemini-2.0-flash-lite
# - models/gemini-exp-1206
# - models/gemini-2.5-flash-preview-tts
# - models/gemini-2.5-pro-preview-tts
# - models/gemma-3-1b-it
# - models/gemma-3-4b-it
# - models/gemma-3-12b-it
# - models/gemma-3-27b-it
# - models/gemma-3n-e4b-it
# - models/gemma-3n-e2b-it
# - models/gemini-flash-latest
# - models/gemini-flash-lite-latest
# - models/gemini-pro-latest
# - models/gemini-2.5-flash-lite
# - models/gemini-2.5-flash-image
# - models/gemini-2.5-flash-preview-09-2025
# - models/gemini-2.5-flash-lite-preview-09-2025
# - models/gemini-3-pro-preview
# - models/gemini-3-flash-preview
# - models/gemini-3-pro-image-preview
# - models/nano-banana-pro-preview
# - models/gemini-robotics-er-1.5-preview
# - models/gemini-2.5-computer-use-preview-10-2025
# - models/deep-research-pro-preview-12-2025