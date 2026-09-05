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
st.caption(
    "Ask questions about International Law. "
    "Answers are grounded only in the provided PDF."
)


# -------------------------------------------------
# Chat Memory
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


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
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        max_output_tokens=1024
    )


vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

model = load_model()


# -------------------------------------------------
# Prompt
# -------------------------------------------------
template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI Assistant specializing in International Law.

Use only the provided retrieved context to answer the user's question.

You may use the conversation history only to understand references,
follow-up questions, and conversational context.

Do not use your general knowledge to provide factual information
that is not supported by the retrieved context.

If the answer cannot be found in the provided context, say:
"I could not find the relevant answer"

Do not make up or assume information."""
    ),
    (
        "human",
        """Conversation history:
{chat_history}

Retrieved context:
{context}

Current question:
{question}"""
    )
])


# -------------------------------------------------
# Display Chat History
# -------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


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
    ask_button = st.button(
        "Ask",
        type="primary",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "Clear",
        use_container_width=True
    )


# -------------------------------------------------
# Clear Conversation
# -------------------------------------------------
if clear_button:
    st.session_state.messages = []
    st.rerun()


# -------------------------------------------------
# Ask Question
# -------------------------------------------------
if ask_button and query.strip():

    # Create conversation history before adding current question
    chat_history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in st.session_state.messages
    )

    with st.spinner(
        "Retrieving relevant context and generating answer..."
    ):

        # -------------------------------------------------
        # Retrieve relevant documents
        # -------------------------------------------------
        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------
        prompt = template.invoke({
            "chat_history": chat_history,
            "context": context,
            "question": query
        })

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------
        result = model.invoke(prompt)

    answer = result.content

    # Handle content blocks if returned by the model
    if isinstance(answer, list):
        answer = "\n".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
        )

    # -------------------------------------------------
    # Save conversation to memory
    # -------------------------------------------------
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # -------------------------------------------------
    # Display current answer
    # -------------------------------------------------
    st.markdown("### Answer")
    st.markdown(answer)

    # -------------------------------------------------
    # Show retrieved context
    # -------------------------------------------------
    with st.expander("Show retrieved context"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i + 1}:**")
            st.markdown(doc.page_content)
            st.markdown("---")


elif ask_button and not query.strip():

    st.warning("Please enter a question.")