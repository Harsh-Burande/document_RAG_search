import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from tavily import TavilyClient
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------------
# ENVIRONMENT SETUP
# -----------------------------------------
# This section loads API keys from the .env file
# and initializes external services used in the project:
# - Gemini (for LLM responses and embeddings)
# - Tavily (for real-time web search fallback)
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TAVILY_API_KEY = os.getenv("TAVILY_KEY")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# FAISS index location
index_path = "faiss_index"

# -----------------------------------------
# DATA PROCESSING
# -----------------------------------------
# This section handles document ingestion and preparation.
# It loads user-provided content from:
# 1. Uploaded PDF or TXT files
# 2. A URL (web page)
# The loaded content is converted into LangChain Document objects.

# Loading documents
def load_document(uploaded_file, url_input):
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp.pdf")

        else:
            with open("temp.txt", "wb") as f:
                f.write(uploaded_file.read())
            loader = TextLoader("temp.txt")
        
        documents = loader.load()

        print(documents[0].page_content[:1000])
    
        return documents
    
    elif url_input:
        loader = WebBaseLoader(url_input)
        docs = loader.load()

        print(docs[0].page_content[:1000])

        return docs

# -----------------------------------------
# DATA PROCESSING – TEXT CHUNKING
# -----------------------------------------
# Large documents cannot be directly processed by embeddings or LLMs.
# This function splits the document into smaller chunks so that:
# - Each chunk can be embedded
# - Retrieval becomes more precise
# - Context fits within model token limits.

# Text splitting and Chunking
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 100
    )
    return text_splitter.split_documents(documents)

# -----------------------------------------
# DATA PROCESSING – RAW TEXT EXTRACTION
# -----------------------------------------
# This function extracts all the text content from the document.
# It is mainly used for summarization, where the entire document
# content needs to be passed to the LLM.

# Extracting Text
def extract_text(documents):
    full_text = ""
    for doc in documents:
        full_text += doc.page_content + "\n"
        
    return full_text

# -----------------------------------------
# REAL-TIME WEB SEARCH (FALLBACK SYSTEM)
# -----------------------------------------
# If the answer cannot be found in the uploaded documents,
# the system falls back to Tavily search to retrieve information
# from the internet. This ensures the assistant still returns
# a useful answer even when the document lacks the information.

# Web-Search function
def web_search(query):

    results = tavily.search(query=query, max_results=3)

    content = "\n\n".join(
        [f"{r['title']}\n{r['content']}\nSource: {r['url']}" for r in results["results"]]
    )

    return content

# -----------------------------------------
# RAG CHAIN (Retrieval Augmented Generation)
# -----------------------------------------
# This is the core logic of the system.
# Steps:
# 1. Retrieve relevant document chunks from the vector database
# 2. Build a prompt containing the retrieved context
# 3. Send the prompt to the Gemini LLM
# 4. If the answer is not found in the document context,
#    trigger web search fallback.

# Generate output to the user query
def answer_question(query, vector_store):

    docs = vector_store.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    model = genai.GenerativeModel("gemini-flash-lite-latest")

    prompt = f"""
            You are a document question answering system.

            Use ONLY the provided context.

            If the answer is not explicitly written in the context, return exactly:
            NOT_FOUND

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

    response = model.generate_content(prompt)
    answer = response.text.strip()

    if "NOT_FOUND" in answer:
        return web_search(query), "web"

    return answer, "document"

# -----------------------------------------
# LLM SUMMARIZATION
# -----------------------------------------
# This function summarizes the entire document.
# It extracts raw text and sends it to Gemini
# to generate a concise summary for quick understanding.

# Summarizing and Generating Output Text
def summarize_text(text):
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    prompt = f"""
    Summarize the following clearly and concisely:

    {text[:4000]}
    """

    response = model.generate_content(prompt)
    return response.text

    
# # # # # # Streamlit Application
# -----------------------------------------
# UI FLOW – STREAMLIT APPLICATION
# -----------------------------------------
# This section defines the user interface.
# The Streamlit app allows users to:
# - Upload documents
# - Provide URLs
# - Ask questions
# - Enable real-time web search
# - Generate document summaries.

# # # # # # Streamlit Application
st.set_page_config(page_title="Smart AI Assistant")
st.title("Multi-Document RAG Search Engine (with Real-Time Web Search)")

# Sidebar
web_toggle = st.sidebar.toggle("Enable Real-Time Web-Search")

# Upload file 
uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

# URL Input
url_input = st.text_input("Or past a URL")

# Query Input
query = st.text_input("Ask your question.")

# Placeholder for output
if query:
    st.write("You asked:", query)
    st.write("Web Search Enebled:", web_toggle)



# -----------------------------------------
# MAIN APPLICATION EXECUTION FLOW
# -----------------------------------------
# Once the user enters a query, the system decides which mode to use:
# 1. Web Mode → Direct web search using Tavily
# 2. Document Mode → RAG pipeline using uploaded documents

if query and len(query.strip()) > 3:

    # WEB MODE (skip document processing)
    if web_toggle:
        with st.spinner("Searching the web..."):
            answer = web_search(query)
            source = "web"

        st.subheader("Answer")
        st.write(answer)

        st.caption("🌐 Source: Web (Tavily)")

    # DOCUMENT MODE
    else:

        if uploaded_file or url_input:

            documents = load_document(uploaded_file, url_input)

            if documents:

                # Split documents
                docs = split_documents(documents)

                # -----------------------------------------
                # EMBEDDINGS
                # -----------------------------------------
                # Convert each document chunk into a numerical vector
                # using Gemini Embedding model. These vectors capture
                # semantic meaning of the text, enabling similarity search.

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=GOOGLE_API_KEY
                )

                # -----------------------------------------
                # VECTOR DATABASE (FAISS)
                # -----------------------------------------
                # FAISS stores embeddings in a vector index so that
                # we can quickly retrieve the most relevant document
                # chunks for a user query using similarity search.

                vector_store = FAISS.from_documents(docs, embeddings)

                # RAG answer
                with st.spinner("Answering..."):
                    answer, source = answer_question(query, vector_store)

                st.subheader("Answer")
                st.write(answer)

                if source == "document":
                    st.caption("📄 Source: Document")
                else:
                    st.caption("🌐 Source: Web (Tavily)")

                # Summary
                if st.button("Generate Summary"):
                    with st.spinner("Summarizing..."):
                        raw_text = extract_text(documents)
                        summary = summarize_text(raw_text)
                        st.subheader("Summary")
                        st.write(summary)







 


# activate virtual environment in terminal

# cd "C:\Users\User1\Desktop\Agentic AI Project" (optional)

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\venv\Scripts\Activate.ps1


# changes into github code.
# git add .
# git commit -m "Added PDF summarization feature"
# git push

# git pull origin main --rebase
# git push