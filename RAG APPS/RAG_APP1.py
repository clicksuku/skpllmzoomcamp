#LlangChain -> PYPDF Parser->PDF Document Chunking
# BGE-Small from HuggingFace Hub->Embeddings
# FAISS Vector Database->Claude Haiku as LLM

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

document_url = "SKPRes.pdf"
loader = PyPDFLoader(document_url)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40, length_function=len, is_separator_regex=False)
chunks = text_splitter.split_documents(pages)

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

chunk_texts =  list(map(lambda d:d.page_content, chunks))
embeddings = model.embed_documents(chunk_texts)

text_embedding_pairs = zip(chunk_texts, embeddings)
db = FAISS.from_embeddings(text_embedding_pairs, model)

query = "what are the domains in the document"
contexts = db.similarity_search(query, k=2)
print(contexts[0])

query = "Awards Certifications"
contexts = db.similarity_search(query, k=2)
print(contexts[0])

