# pip install sentence_transformers PyPDF langchain langchain_community
# pip install streamlit
# To install ollama - Downloaded the Application from the site and installed Command line tools
# Milvus Client was very ok in Vector Search
# Trying Pinecone now

import os
from dotenv import load_dotenv
import hashlib

import streamlit as st

import openai
from openai import OpenAI

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone

system = """
You are a bot who answers based on the context provided. 
- REPLY IN A FRIENDLY TONE.
- IF YOU DON'T KNOW ANSWER 'I DO NOT KNOW. PLEASE REFER THE DOCUMENT'.
Begin the conversation with a warm greeting.
At the end of the conversation, respond with "<|DONE|>"."""


load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")    
pc = Pinecone(api_key=pinecone_key)
index = pc.Index("skpllmindex2")

OpenAI_Key = os.getenv("OPENAI_API_KEY")

def create_embeddings(chunks:list[Document]):
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    huggingFaceEmbedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    
    embeddings = huggingFaceEmbedding.embed_documents(chunks)
    return embeddings

def generate_short_id(content: str) -> str:
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))
    return hash_obj.hexdigest()


def combine_vector_and_text(
        documents: list[any], doc_embeddings: list[list[float]]
    ) -> list[dict[str, any]]:
    data_with_metadata = []

    for doc_text, embedding in zip(documents, doc_embeddings):
        # Convert doc_text to string if it's not already a string
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        # Generate a unique ID based on the text content
        doc_id = generate_short_id(doc_text)

        data_item = {
            "id": doc_id,
            "values": embedding,
            "metadata": {"text": doc_text},  # Include the text as metadata
        }
        data_with_metadata.append(data_item)
    return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
    index.upsert(vectors=data_with_metadata)

def create_embeddings_for_query(query)->list[float]:
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

def query_pinecone_index(
                query_embeddings: list, top_k: int = 5, include_metadata: bool = True
            ) -> dict[str, any]:
    query_response = index.query(
            vector=query_embeddings, top_k=top_k, include_metadata=include_metadata
        )
    return query_response

#change it to read from URL into Stream and read from stream here
def process_documents():
    st.write("Entered Processing Documents")
    loader = PyPDFDirectoryLoader("./")
    docs = loader.load()
    txt_splitters = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=40,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

    return txt_splitters.split_documents(docs)

def chat_with_llm(context_documents,query):
    values = context_documents["matches"]
    context = "\n".join([value['metadata']['text'] for value in values])
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f'The context is {context}'}
    ]
    st.write(context)    
    prompt = f"Answer the following question based on the provided context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    messages.append(
        {"role": "user", "content": prompt},
    )
    client = OpenAI(api_key=OpenAI_Key)

    # Request response from OpenAI LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or any other available OpenAI LLM models
        messages=messages,
    )
    #return response['choices'][0]
    return response


if __name__== "__main__":
    st.set_page_config(page_title="RAG and Local LLM on Sundarkp's Resume")
    st.header("A simple attempt to leverage rag")

    collection_name="SKP_Profile_Collection"
    embeddings = ""
    chunks = ""
    chunk_texts = ""
    query = ""

    with st.sidebar:
        st.header("Press the Process Button to load my profile into rag\n")
        process = st.button("Process")
        query = st.text_input("Query : ", "What domains he has worked on")
        ask = st.button("Query")

    if(process):
        chunks = process_documents() #Created document chunks
        chunk_texts =  list(map(lambda d:d.page_content, chunks))
        embeddings = create_embeddings(chunk_texts)
        data_with_meta_data = combine_vector_and_text(chunk_texts, embeddings) 
        upsert_data_to_pinecone(data_with_metadata= data_with_meta_data)
        st.write("Data Inserted into Vector DB")
    
    if(ask):
        query_embeddings = create_embeddings_for_query(query)
        query_response = query_pinecone_index(query_embeddings=query_embeddings)
        st.write(query_response)
        response = chat_with_llm(query_response,query)
        st.write(response)