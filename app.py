import streamlit as st
import tempfile
import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Document Q&A Bot using RAG", layout="wide")
st.title("RAG-Docs")

if "messages" not in st.session_state:
    st.session_state.messages = []  # To store chat history

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None 

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Processing the uploaded files
def process_files(files):
    all_docs = []
    
    # Looping through files
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
        
        try:
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Splitting the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    # Hybrid Search
    embeddings = OllamaEmbeddings(model="bge-m3:latest")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Sparse Retriever
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 2

    # Combining them
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever

# Sidebar to upload files
with st.sidebar:
    st.header("Data Source")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Chunking, Embedding, and Building Hybrid Index..."):
            st.session_state.retriever = process_files(uploaded_files)
            st.success("Knowledge Base Ready!")

    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()


# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Inupt
if user_input := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    if st.session_state.retriever is None:
        st.error("Please upload and process documents first.")
    else:
        llm = ChatOllama(model="alibayram/smollm3:latest", temperature=0.2, num_gpu=1)

        # History Aware Retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, st.session_state.retriever, contextualize_q_prompt
        )

        # Answering the question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Keep the answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Streaming the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            source_documents = []

            # Convert session history to LangChain format
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))

            with st.spinner("Thinking..."):
                response_stream = rag_chain.stream({
                    "input": user_input,
                    "chat_history": chat_history
                })

                for chunk in response_stream:
                    if "answer" in chunk:
                        full_response += chunk["answer"]
                        message_placeholder.markdown(full_response + "▌")
                    if "context" in chunk:
                        source_documents = chunk["context"]

                message_placeholder.markdown(full_response)

                # Show citations
                if source_documents:
                    with st.expander("Source Citations"):
                        for i, doc in enumerate(source_documents):
                            page_num = doc.metadata.get('page', 'Unknown')
                            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            snippet = doc.page_content[:150].replace("\n", " ")
                            st.markdown(f"**Source {i+1}** (File: {source_file}, Page: {page_num})")
                            st.caption(f"\"{snippet}...\"")

            # Saving to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})