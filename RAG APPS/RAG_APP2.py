# pip install sentence_transformers PyPDF langchain langchain_community
# pip install streamlit
# To install ollama - Downloaded the Application from the site and installed Command line tools


import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from pymilvus import MilvusClient


def create_vector_db():
    client = MilvusClient(uri="./skp_profile_llm.db")    
    return client

def create_vector_collection(client, collection_name, chunks,embeddings):
    if(client.has_collection(collection_name)):
        client.drop_collection(collection_name)
    
    client.create_collection(
            collection_name=collection_name,
            dimension=384,
        )
    
    chunk_texts =  list(map(lambda d:d.page_content, chunks))

    data = [{"id":i, "vector":embeddings[i], "text":chunk_texts[i]} for i in range(len(embeddings))]
    collection = client.insert(collection_name=collection_name,data=data)
    return collection

def create_embeddings_from_text(query):
    print("Creating Embeddings")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    huggingFaceEmbedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    query_embeddings = huggingFaceEmbedding.embed_query(query)
    return query_embeddings

def create_embeddings(chunks:list[Document]):
    print("Creating Embeddings")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    huggingFaceEmbedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    chunk_texts =  list(map(lambda d:d.page_content, chunks))
    embeddings = huggingFaceEmbedding.embed_documents(chunk_texts)
    return embeddings
    

#change it to read from URL into Stream and read from stream here
def process_documents():
    st.write("Entered Processing Documents")
    loader = PyPDFDirectoryLoader("./")
    docs = loader.load()
    for doc in docs:
        st.write(doc.metadata["source"])
        #st.write(doc.page_content)

    txt_splitters = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=40,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

    return txt_splitters.split_documents(docs)

if __name__== "__main__":
    st.set_page_config(page_title="RAG and Local LLM on Sundarkp's Resume")
    st.header("A simple attempt to leverage rag")

    collection_name="SKP_Profile_Collection"
    embeddings = ""
    chunks = ""

    with st.sidebar:
        st.header("Press the Process Button to load my profile into rag\n")
        process = st.button("Process")
    
    if(process):
        chunks = process_documents() #Created document chunks
        embeddings = create_embeddings(chunks)
        #st.write(embeddings)

    client=create_vector_db()
    collection = create_vector_collection(client, collection_name, chunks,embeddings)

    st.write("Going into Search Mode now")
    query = "What are the domains worked on?"
    query_embeddings = create_embeddings_from_text(query)

    response = client.query(
        collection_name=collection_name,
        data=query_embeddings,
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    st.write(response)