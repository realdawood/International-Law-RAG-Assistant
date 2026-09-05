import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from embedding.embeddings import embedding_model

load_dotenv()

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="International Law RAG Assistant",
    page_icon="⚖️",
    layout="centered",
)

st.title("⚖️ International Law Q&A")
st.caption("Ask questions about International Law. Answers are grounded only in the provided PDF.")

# -------------------------------------------------
# Load resources once (cached)
# -------------------------------------------------
@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(model = "gemini-3.6-flash", max_output_tokens = 1024)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
)
model = load_model()

template = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a helpful AI Assistant.
Use only the provided context to answer the question.
If you do not find the answer in the context, say "I could not find the relevant answer" and do not make up an answer.
"""),
    ("human", """context : {context},
question : {question}""")
])

# -------------------------------------------------
# Sidebar – settings
# -------------------------------------------------
    

# -------------------------------------------------
# Main chat interface
# -------------------------------------------------
query = st.text_area(
    "Your question about International Law:",
    height=100,
    placeholder="e.g. What are the main sources of international law?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.rerun()

if ask_button and query.strip():
    with st.spinner("Retrieving relevant context and generating answer..."):
        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build prompt & call model
        prompt = template.invoke({"context": context, "question": query})
        result = model.invoke(prompt)

    # Display answer
    st.markdown("### Answer")
    st.markdown(result.content[0]["text"])

    # Optional: show retrieved chunks
    with st.expander("Show retrieved context"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.markdown(doc.page_content)
            st.markdown("---")

elif ask_button and not query.strip():
    st.warning("Please enter a question.")