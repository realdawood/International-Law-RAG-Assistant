from dotenv import load_dotenv
from langchain.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from embedding.embeddings import embedding_model


load_dotenv()  # Load environment variables from .env file

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

retriever = vectorstore.as_retriever(search_type =  'mmr', search_kwargs = {'k' : 3, 'fetch_k' : 10, 'lambda_mult' : 0.5})

model = ChatGoogleGenerativeAI(model = "gemini-3.6-flash", max_output_tokens = 1024)

template = ChatPromptTemplate.from_messages([("system", 
"""You are a helpful AI Assitant.
Use only the provided context to answer the question.
if you do not find the answer in the context, say "I could not find the relevant answer" and do not make up an answer.
"""), ("human", """context : {context},
question : {question}""")])



query = input("Feel free to ask any question related to International Law: ")

docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in docs])

prompt = template.invoke({"context": context, "question": query})


result = model.invoke(prompt)

print(result.content[0]["text"])

