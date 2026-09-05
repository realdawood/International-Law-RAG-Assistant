from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_NAME
)