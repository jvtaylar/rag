import streamlit as st
import openai
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --------------------------
# Azure OpenAI Configuration
# --------------------------
AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/"
AZURE_OPENAI_KEY = "<your-key>"
AZURE_DEPLOYMENT_NAME = "<your-deployment-name>"   # example: gpt-35-turbo
AZURE_EMBEDDING_NAME = "<your-embedding-deployment>"  # example: text-embedding-ada-002

openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_KEY
openai.api_version = "2024-02-01"

# --------------------------
# Load Documents (Sample FAQ/Manuals)
# --------------------------
docs = [
    Document(page_content="TESDA provides technical vocational education and training in the Philippines."),
    Document(page_content="TESDA offers assessment and certification for skilled workers."),
    Document(page_content="You can apply for TESDA scholarships through their official website."),
]

# --------------------------
# Split and Embed Documents
# --------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

embeddings = AzureOpenAIEmbeddings(
    model=AZURE_EMBEDDING_NAME,
    azure_deployment=AZURE_EMBEDDING_NAME,
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT
)

vectordb = Chroma.from_documents(split_docs, embeddings)

retriever = vectordb.as_retriever()

# --------------------------
# Define LLM (Azure OpenAI)
# --------------------------
llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    openai_api_version="2024-02-01",
    openai_api_key=AZURE_OPENAI_KEY,
    openai_api_base=AZURE_OPENAI_ENDPOINT,
    temperature=0
)

# --------------------------
# Create RetrievalQA Chain
# --------------------------
prompt_template = """
You are a helpful assistant. Use the following documents to answer the question.

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# --------------------------
# Streamlit UI
# --------------------------
st.title("💬 TESDA RAG Chatbot (Azure OpenAI + Streamlit)")

user_query = st.text_input("Ask me anything about TESDA:")

if user_query:
    response = qa_chain.run(user_query)
    st.markdown(f"**Answer:** {response}")
