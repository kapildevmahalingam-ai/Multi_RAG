import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

# 1. Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Multi-PDF RAG App", page_icon="📚", layout="centered"
)

st.title("📚 Multi-PDF RAG App with Sources")
st.markdown(
    "Upload multiple PDF documents below to index them into a single vector"
    " database and search across all of them."
)

# 2. File uploader for multiple PDFs (matching assignment specs)
files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

# 3. Indexing logic stored in session state
if files and "store" not in st.session_state:
  with st.spinner("Processing and indexing uploaded documents..."):
    chunks = []
    for f in files:
      # Save uploaded file to a temporary location so PyPDFLoader can read it
      with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(f.getvalue())
        tmp_path = tmp_file.name

      loader = PyPDFLoader(tmp_path)
      pages = loader.load()

      # Carry filename through metadata
      for page in pages:
        page.metadata["source"] = f.name

      # Split text into chunks
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500, chunk_overlap=50
      )
      file_chunks = text_splitter.split_documents(pages)
      chunks.extend(file_chunks)

      # Clean up temporary file
      os.unlink(tmp_path)

    # Embed and create a single FAISS vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    st.session_state.store = FAISS.from_documents(chunks, embeddings)

    st.success(
        f"Uploaded: {', '.join([f.name for f in files])}\n\nIndexed"
        f" {len(chunks)} chunks."
    )

# 4. Question Answering & Source Tracking Interface
if "store" in st.session_state:
  st.markdown("---")
  question = st.text_input(
      "Q: Ask a question across your documents...",
      placeholder="e.g., What are the main topics?",
  )

  if question:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)

    # Retrieve matching chunks
    retriever = st.session_state.store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    # Compile context
    context_text = "\n\n".join([d.page_content for d in docs])

    # Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert assistant. Answer the user's question using"
                " ONLY the provided context. If the answer cannot be found in"
                " the documents, say 'I cannot find that information in the"
                " documents.'\n\nContext:\n{context}"
            ),
        ),
        ("user", "{input}"),
    ])

    chain = prompt_template | llm

    with st.spinner("Searching documents..."):
      response = chain.invoke({"context": context_text, "input": question})

      if isinstance(response.content, list):
        answer_text = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        )
      else:
        answer_text = response.content

    # Display Answer
    st.markdown("### Answer")
    st.write(answer_text)

    # Display Sources (File name and page number)
    st.markdown("### Sources")
    sources_seen = set()
    for d in docs:
      src = d.metadata.get("source", "Unknown")
      page = (
          d.metadata.get("page", 0) + 1
      )  # 1-indexed page numbering for readability
      source_identifier = (src, page)

      if source_identifier not in sources_seen:
        sources_seen.add(source_identifier)
        st.caption(f"📄 **{src}** — page {page}")