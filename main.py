import os
import streamlit as st
import pickle
import time

from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

st.title("Movie Review Analysis Tool")
st.sidebar.title("Movie Review URLs")

urls=[]
for i in range(4):
    url = st.sidebar.text_input(f"URL {i}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process")

file_path = "vector_index.pkl"
main_placefolder = st.empty()

if process_url_clicked:
    # Init LLM
    llm = OpenAI(temperature=0.9, max_tokens=500)

    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Loading...")
    data = loader.load()

    # Split data
    main_placefolder.text("Splitting data...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    # Make embeddings and save
    main_placefolder.text("Saving embeddings...")
    embeddings = OpenAIEmbeddings()
    vectorindex = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    
    # Save 'vector database' locally
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)

query = main_placefolder.text_input("Question: ")
if query:
    if os.path.exists(file_path) and llm:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            # Display sources
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources consulted:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
