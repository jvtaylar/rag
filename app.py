import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# --- Azure OpenAI Configuration ---
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01" # Or your desired version
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# --- Streamlit UI ---
st.set_page_config(page_title="Azure OpenAI RAG Chatbot", page_icon="🤖")
st.title("🤖 Azure OpenAI RAG Chatbot")

# Initialize chat history in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# --- Functions for document processing and RAG setup ---

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )

@st.cache_resource(show_spinner=False)
def get_llm_model():
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        temperature=0.7,
    )

def get_document_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks, embeddings_model):
    with st.spinner("Creating knowledge base..."):
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings_model)
    st.success("Knowledge base created!")
    return vectorstore

def get_conversation_chain(vectorstore, llm_model):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- File Uploader ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF or TXT files here and click 'Process'",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    process_button = st.button("Process Documents")

    if process_button and uploaded_files:
        raw_text = ""
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == "pdf":
                loader = PyPDFLoader(uploaded_file.name)
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                docs = loader.load()
                for doc in docs:
                    raw_text += doc.page_content
                os.remove(uploaded_file.name) # Clean up temp file
            elif file_extension == "txt":
                raw_text += uploaded_file.read().decode("utf-8")

        if raw_text:
            text_chunks = get_document_chunks(raw_text)
            embeddings = get_embeddings_model()
            st.session_state.vectorstore = create_vectorstore(text_chunks, embeddings)
            llm = get_llm_model()
            st.session_state.conversation_chain = get_conversation_chain(st.session_state.vectorstore, llm)
            st.session_state.messages.append({"role": "assistant", "content": "Documents processed! You can now ask questions."})
            st.rerun() # Rerun to update the chat UI with the new message
        else:
            st.warning("No text extracted from the uploaded files.")
    elif process_button and not uploaded_files:
        st.warning("Please upload some files first!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input and Response ---
if prompt := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.conversation_chain:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation_chain({'question': prompt})
            st.session_state.chat_history = response['chat_history']
            bot_response = response['answer']

        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.chat_message("assistant"):
            st.markdown("Please upload and process documents first in the sidebar.")
        st.session_state.messages.app
