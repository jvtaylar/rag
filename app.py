import streamlit as st
import openai
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --------------------------
# Azure OpenAI Configuration
# --------------------------
openai.api_type = "azure"
openai.api_base = "https://jvtay-mff428jo-eastus2.openai.azure.com/"
openai.api_version = "2025-01-01-preview"
openai.api_key = "FOObvelUv1Ubbw0ZlEb3NPCBYDbdXWbLhzyckQAA9cP3Ofhgi8KWJQQJ99BIACHYHv6XJ3w3AAAAACOGoHUz"   # 🔑 put your key here

DEPLOYMENT_NAME = "gpt-35-turbo"
EMBEDDING_NAME = "text-embedding-ada-002"

# --------------------------
# Example documents
# --------------------------
docs = [
    Document(page_content="TESDA provides technical vocational education and training in the Philippines."),
    Document(page_content="TESDA offers assessment and certification for skilled workers."),
    Document(page_content="You can apply for TESDA scholarships through their official website."),
]

# --------------------------
# Split & store docs in vector DB
# --------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = AzureOpenAIEmbeddings(
    model=EMBEDDING_NAME,
    azure_deployment=EMBEDDING_NAME
)

vectordb = Chroma.from_documents(split_docs, embeddings)
retriever = vectordb.as_retriever()

# --------------------------
# LLM (Azure OpenAI)
# --------------------------
llm = AzureChatOpenAI(
    azure_deployment=DEPLOYMENT_NAME,
    temperature=0
)

# --------------------------
# RetrievalQA
# --------------------------
prompt_template = """
Use the context below to answer the question.

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

query = st.text_input("Ask a question:")

if query:
    answer = qa_chain.run(query)
    st.write("**Answer:**", answer)
