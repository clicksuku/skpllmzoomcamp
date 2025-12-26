# pip install sentence_transformers PyPDF langchain langchain_community
# pip install streamlit
# To install ollama - Downloaded the Application from the site and installed Command line tools
# Milvus Client was very ok in Vector Search
# Trying Pinecone now

import hashlib
import streamlit as st

from openai import OpenAI

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore

system = """
You are a bot who answers based on the context provided. 
- REPLY IN A FRIENDLY TONE.
- IF YOU DON'T KNOW ANSWER 'I DO NOT KNOW. PLEASE REFER THE DOCUMENT'.
Begin the conversation with a warm greeting.
At the end of the conversation, respond with 
"For more details, refer to my profile at https://www.linkedin.com/in/sundarkp/"."""

OpenAI_Key = st.secrets["OPENAI_API_KEY"]

huggingFaceEmbeddingmodel = ""

class FaissDb(object):
    def __init__(self,chunk_texts, embeddings):
        text_embedding_pairs = zip(chunk_texts, embeddings)
        self.faissDb = FAISS.from_embeddings(text_embedding_pairs, huggingFaceEmbeddingmodel)

    def query_faiss(self, prompt): 
        query_response = self.faissDb.similarity_search(prompt, k=2)
        return query_response
    
    def get_faiss(self):
        return self.faissDb


def create_embeddings(chunks:list[Document]):
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    huggingFaceEmbeddingmodel = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )  
    embeddings = huggingFaceEmbeddingmodel.embed_documents(chunks)
    return embeddings

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
    #st.write(context)    
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

    chunks = process_documents() #Created document chunks
    chunk_texts =  list(map(lambda d:d.page_content, chunks))
    embeddings = create_embeddings(chunk_texts)
    faissDb = FaissDb(chunk_texts, embeddings)
    st.write("Data Inserted into Vector DB")
    query = ""

    with st.sidebar:
        #st.header("Press the Process Button to load my profile into rag\n")
        #process = st.button("Process")
        query = st.text_input("Query : ", "What domains he has worked on")
        ask = st.button("Query")

    if(ask):
        query_response = faissDb.query_faiss(query)
        st.write(query_response)
        #response = chat_with_llm(query_response,query)
        #for choice in response.choices:
            #st.write(choice.message.content)