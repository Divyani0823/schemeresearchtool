import streamlit as st
import openai
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import os
import requests
import fitz  # PyMuPDF
import faiss
import numpy as np
import json

# Load API Key and set it as an environment variable
def load_api_key():
    try:
        with open('.config', 'r') as f:
            api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = api_key  # Set the environment variable
            return api_key
    except FileNotFoundError:
        st.error("API key file (.config) not found.")
        return None

# Load API Key
api_key = load_api_key()
if not api_key:
    st.error("OpenAI API key not found!")
    st.stop()

# Function to download and extract text from PDF URL
def extract_text_from_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP status codes

        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        
        doc = fitz.open("temp.pdf")
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        return text
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to extract key scheme details
def extract_scheme_details(text):
    prompt = f"""
    Analyze the following text and extract key information about the scheme. Provide the information in the following format:
    1. Scheme Benefits:
    2. Scheme Application Process:
    3. Eligibility Criteria:
    4. Documents Required:

    Text:
    {text}
    """
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")  # Use ChatOpenAI for GPT-4
    response = llm.predict(prompt)
    return response

# Function to process URLs (including PDFs)
def process_urls(urls):
    if not urls:
        st.error("No URLs provided.")
        return None, None, None

    all_docs = []
    summarized_data = []

    for url in urls:
        if url.endswith(".pdf"):
            text = extract_text_from_pdf(url)
            if text:
                all_docs.append(Document(page_content=text, metadata={"source": url}))
                # Extract key scheme details
                scheme_details = extract_scheme_details(text)
                summarized_data.append({
                    "source": url,
                    "summary": scheme_details
                })
        else:
            loader = UnstructuredURLLoader([url])
            try:
                docs = loader.load()
                if docs:
                    all_docs.extend(docs)
                    for doc in docs:
                        scheme_details = extract_scheme_details(doc.page_content)
                        summarized_data.append({
                            "source": url,
                            "summary": scheme_details
                        })
            except Exception as e:
                st.error(f"Error loading URL: {e}")
    
    if not all_docs:
        st.error("No documents loaded from the provided URLs.")
        return None, None, None

    st.success(f"Loaded {len(all_docs)} documents from URLs.")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = splitter.split_documents(all_docs)
    
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = FAISS.from_documents(docs_split, embeddings)
        faiss.write_index(vector_store.index, "faiss_store_openai.index")

        # Save summaries and doc mapping
        st.session_state.summarized_data = summarized_data
        doc_mapping = {i: doc for i, doc in enumerate(docs_split)}
        st.session_state.doc_mapping = doc_mapping

        # Save summarized data locally
        with open("summarized_data.json", "w") as f:
            json.dump(summarized_data, f)

    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, None, None
    
    return vector_store, embeddings

# Load existing FAISS index
def load_faiss_index():
    if os.path.exists("faiss_store_openai.index"):
        index = faiss.read_index("faiss_store_openai.index")
        return index
    return None

# Load summarized data
def load_summarized_data():
    if os.path.exists("summarized_data.json"):
        with open("summarized_data.json", "r") as f:
            st.session_state.summarized_data = json.load(f)

# Query the vector store
def query_vector_store(index, embeddings, query):
    try:
        # Embed the query for vector search
        query_embedding = embeddings.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = index.search(query_embedding, k=1)  # Find the most similar document

        if len(indices[0]) > 0 and indices[0][0] != -1:
            doc_id = indices[0][0]
            doc_mapping = st.session_state.get("doc_mapping")
            
            if doc_mapping and doc_id in doc_mapping:
                # Retrieve the relevant document
                result_doc = doc_mapping[doc_id]
                relevant_content = result_doc.page_content

                # Use LLM to generate a query-specific response
                prompt = f"""
                You are an assistant. Use the following text to answer the user's query. Be specific and concise.
                User Query: {query}
                Relevant Text: {relevant_content}
                Answer:
                """
                llm = ChatOpenAI(temperature=0.7, model="gpt-4")  # Use ChatOpenAI for GPT-4
                response = llm.predict(prompt)
                return response
            else:
                return "Relevant document not found in the mapping."
        else:
            return "No similar document found in the vector store."
    except Exception as e:
        st.error(f"Error querying vector store: {e}")
        return None

# Streamlit App Configuration
st.set_page_config(
    page_title="Scheme Research Tool",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("📚 Scheme Research Tool")
st.markdown(
    """
    Welcome to the **Scheme Research Tool**! This application helps you extract, summarize, and search for details about government schemes and programs from URLs and PDFs.
    """
)
st.divider()

# Sidebar Input Options
st.sidebar.title("🔧 Configuration")
st.sidebar.header("📂 Input Options")
input_option = st.sidebar.radio("Choose Input Type:", ("Enter URLs", "Upload URL File"), key="input_option_radio")

if input_option == "Enter URLs":
    urls = st.sidebar.text_area("Enter URLs (one per line):", placeholder="https://example.com/scheme1").splitlines()
elif input_option == "Upload URL File":
    uploaded_file = st.sidebar.file_uploader("Upload Text File with URLs", type=["txt"], key="upload_file_input")
    if uploaded_file:
        urls = uploaded_file.read().decode("utf-8", errors='ignore').splitlines()

# Process URLs Button
if st.sidebar.button("🚀 Process URLs", key="process_urls_button"):
    if urls:
        with st.spinner("Processing URLs... This may take a while."):
            try:
                vector_store, embeddings = process_urls(urls)
                if vector_store:
                    st.sidebar.success("✅ URLs processed and FAISS index created!")
            except ValueError as e:
                st.sidebar.error(f"Error processing URLs: {e}")
    else:
        st.sidebar.error("No URLs provided!")

# Load existing FAISS index
index = load_faiss_index()
if index:
    load_summarized_data()
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Query Section
    st.header("🔍 Query Section")
    st.write("Ask a question related to the schemes and get a detailed answer.")
    query = st.text_input(
        "Enter your query below:",
        placeholder="What are the benefits of the XYZ scheme?",
        key="query_input"
    )
    if st.button("Search", key="search_button"):
        with st.spinner("Searching for the best answer..."):
            answer = query_vector_store(index, embeddings, query)
            st.write("### 📝 Answer:")
            st.success(answer)

# Display Summarized Scheme Details
if "summarized_data" in st.session_state:
    st.header("📄 Summarized Scheme Details")
    st.markdown(
        """
        Below is the extracted information for each scheme. Use these summaries for quick reference.
        """
    )
    for i, data in enumerate(st.session_state.summarized_data, start=1):
        st.subheader(f"🔗 Scheme {i}: {data['source']}")
        st.markdown(f"**Scheme Summary:**\n\n{data['summary']}")
else:
    st.info("No scheme details available. Please process some URLs to get started.")

# Footer
st.divider()
st.markdown(
    """
    **Developed by Divyani** | Powered by OpenAI GPT-4 | Made with ❤️ and Streamlit
    """
)
