import os
import chainlit as cl
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Load documents
loader = TextLoader("docs/faq.txt")
docs = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Create embeddings & vectorstore
embeddings = OpenAIEmbeddings()
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
