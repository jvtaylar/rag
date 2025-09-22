

import streamlit as st
import openai
import os
import tempfile
from io import BytesIO
from typing import List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
import PyPDF2

# -----------------------------
# Helper functions
# -----------------------------

def set_azure_openai(api_key: str, api_base: str, api_version: str = "2023-05-15"):
    """Configure the openai client to use Azure OpenAI."""
    openai.api_type = "azure"
    openai.api_key = api_key
    openai.api_base = api_base
    openai.api_version = api_version


def pdf_to_text(file: BytesIO) -> str:
    try:
        reader = PyPDF2.PdfReader(file)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
        return ""


def txt_to_text(file: BytesIO) -> str:
    try:
        return file.getvalue().decode(errors='ignore')
    except Exception as e:
        st.error(f"Failed to read txt: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def get_embeddings(texts: List[str], engine: str) -> List[List[float]]:
    # Azure OpenAI embedding call
    if not texts:
        return []
    # The API supports batching multiple inputs
    try:
        resp = openai.Embedding.create(input=texts, engine=engine)
        embeddings = [d['embedding'] for d in resp['data']]
        return embeddings
    except Exception as e:
        st.error(f"Embedding request failed: {e}")
        return []


class InMemoryVectorStore:
    def __init__(self):
        self.embeddings = None  # numpy array (n_samples, dim)
        self.chunks = []
        self.metadata = []
        self.nn = None

    def add(self, chunk_texts: List[str], chunk_embeddings: List[List[float]], metadatas: List[dict] = None):
        if not chunk_texts:
            return
        arr = np.array(chunk_embeddings)
        if self.embeddings is None:
            self.embeddings = arr
        else:
            self.embeddings = np.vstack([self.embeddings, arr])
        self.chunks.extend(chunk_texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in chunk_texts])
        # fit neighbor index
        self.nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.nn.fit(self.embeddings)

    def query(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[str, float]]:
        if self.embeddings is None or self.nn is None:
            return []
        dist, idx = self.nn.kneighbors([query_embedding], n_neighbors=min(top_k, len(self.chunks)))
        results = []
        for d, i in zip(dist[0], idx[0]):
            results.append((self.chunks[int(i)], float(d)))
        return results


def create_prompt(context_chunks: List[str], question: str) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Use the provided context to answer the question.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\n\nIf the answer is not contained in the context, say 'I don't know based on the provided documents.'"
    )
    return prompt


def ask_chat_model(deployment: str, prompt: str, max_tokens: int = 400, temperature: float = 0.0):
    try:
        resp = openai.ChatCompletion.create(
            engine=deployment,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Chat request failed: {e}")
        return ""


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Simple RAG — Streamlit + Azure OpenAI")
    st.title("📚 Simple RAG Chatbot (Streamlit + Azure OpenAI)")

    st.sidebar.header("Azure OpenAI Settings")
    api_key = st.sidebar.text_input("Azure OpenAI API Key", type="password")
    api_base = st.sidebar.text_input("Azure OpenAI Base URL (e.g. https://YOUR-RESOURCE.openai.azure.com)")
    api_version = st.sidebar.text_input("Azure OpenAI API Version", value="2023-05-15")
    embeddings_deployment = st.sidebar.text_input("Embeddings deployment name", value="text-embedding-3-small")
    chat_deployment = st.sidebar.text_input("Chat/Completion deployment name", value="gpt-4o-mini")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload PDF or TXT files to build your knowledge base. The app stores vectors in memory (not persistent).")

    if api_key and api_base:
        set_azure_openai(api_key, api_base, api_version)
        st.sidebar.success("Azure OpenAI configured")
    else:
        st.sidebar.info("Enter Azure OpenAI key + base to enable embeddings and chat")

    # file uploader
    uploaded_files = st.file_uploader("Upload PDFs or TXT files", accept_multiple_files=True, type=['pdf', 'txt'])

    # stateful store
    if 'store' not in st.session_state:
        st.session_state['store'] = InMemoryVectorStore()

    if uploaded_files and api_key and api_base:
        with st.spinner("Processing files and creating embeddings — this may take a little while"):
            for uploaded in uploaded_files:
                fname = uploaded.name
                raw = uploaded.read()
                bio = BytesIO(raw)
                if fname.lower().endswith('.pdf'):
                    text = pdf_to_text(bio)
                else:
                    text = txt_to_text(bio)
                if not text:
                    continue
                chunks = chunk_text(text)
                embeddings = get_embeddings(chunks, engine=embeddings_deployment)
                metadatas = [{'source': fname} for _ in chunks]
                st.session_state['store'].add(chunks, embeddings, metadatas)
            st.success("Finished processing uploaded files — vectors stored in memory.")
    elif uploaded_files and (not api_key or not api_base):
        st.warning("Please provide Azure API Key and Base URL in the sidebar before uploading files.")

    st.markdown("---")
    st.header("Ask questions — retrieval augmented answers")

    user_question = st.text_input("Your question")
    top_k = st.slider("Number of context chunks to retrieve", min_value=1, max_value=10, value=3)

    if st.button("Ask"):
        if not user_question:
            st.info("Type a question first.")
        elif st.session_state['store'].embeddings is None:
            st.info("Upload files first to build the knowledge base.")
        else:
            with st.spinner("Retrieving relevant chunks and querying the model..."):
                q_emb = get_embeddings([user_question], engine=embeddings_deployment)
                if not q_emb:
                    st.error("Failed to create query embedding.")
                else:
                    results = st.session_state['store'].query(q_emb[0], top_k=top_k)
                    if not results:
                        st.info("No context available. Try uploading documents first.")
                    else:
                        context_chunks = [r[0] for r in results]
                        prompt = create_prompt(context_chunks, user_question)
                        answer = ask_chat_model(chat_deployment, prompt)

                        st.subheader("Answer")
                        st.write(answer)

                        st.subheader("Retrieved chunks (for transparency)")
                        for i, (chunk, dist) in enumerate(results):
                            st.markdown(f"**Chunk {i+1}** — distance: {dist:.4f}")
                            st.write(chunk[:1000] + ("..." if len(chunk) > 1000 else ""))

    st.markdown("---")
    st.caption("Notes: This is a simple demo. For production use you should: persist vectors to a database, add deduplication, improve chunking & prompt engineering, rate limit and secure keys, and add error handling.")


if __name__ == '__main__':
    main()
