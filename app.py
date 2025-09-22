import os
import chainlit as cl
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Azure OpenAI settings
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    openai_api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    deployment_name=azure_deployment,
    openai_api_version=azure_api_version,
    temperature=0,
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=azure_api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version=azure_api_version,
)

# Load documents
loader = TextLoader("docs/faq.txt")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Create embeddings & vectorstore
vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")

# Retrieval-based QA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

@cl.on_chat_start
async def start():
    await cl.Message(content="👋 Hello! Ask me anything from the FAQs.").send()

@cl.on_message
async def main(message: str):
    response = qa.run(message)
    await cl.Message(content=response).send()
