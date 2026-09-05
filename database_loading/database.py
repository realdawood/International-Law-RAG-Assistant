from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding.embeddings import embedding_model
from langchain_chroma import Chroma

data = PyPDFLoader('assets/base_document/international-law_compress.pdf')

docs = data.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectors = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory="chroma_db")