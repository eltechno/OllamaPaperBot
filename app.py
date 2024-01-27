import streamlit as st #
import os
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama

# from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

#This is the fastest of the PDF parsing options, and contains detailed metadata about the PDF and its pages, as well as returns one document per page.
#from langchain_community.document_loaders import PyMuPDFLoader


# RAG prompt
from langchain import hub

# prompt = hub.pull("rlm/rag-prompt")
# prompt = hub.pull("rlm/rag-prompt-llama")
prompt = hub.pull("rlm/rag-prompt-mistral")


#Embedding Size
# 'thenlper/gte-base': 768,
# 'thenlper/gte-large': 1024,
# 'BAAI/bge-large-en': 1024,
# 'text-embedding-ada-002': 1536,
# 'gte-large-fine-tuned': 1024


#model_path = "sentence-transformers/all-MiniLM-L6-v2"
#model_path = "sentence-transformers/all-MiniLM-L12-v2"
#model_path = "BAAI/bge-large-en-v1.5"
model_path = "BAAI/llm-embedder" # Load model automatically use GPUs


vectorstore_directory = "vectorstore_data"
file_directory = "files"
# llmmodel="solar:10.7b"
# llmmodel="openchat:7b-v3.5"
# llmmodel="dolphin-mixtral:8x7b"
# llmmodel="starling-lm"
llmmodel = "mistral:latest"

# Ensure the vectorstore directory exists
if not os.path.exists(vectorstore_directory):
    os.makedirs(vectorstore_directory)

if not os.path.exists(file_directory):
    os.makedirs(file_directory)

# if "template" not in st.session_state:
#     #st.session_state.template = """ Your tone should be professional and informative. Answer the question based only on the following context:
#     st.session_state.template = """ You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use as much sentences to provide the best and concise the answer.
#     Context: {context}
#     History: {history}

#     User: {question}
#     Chatbot:"""

# if "prompt" not in st.session_state:
#     st.session_state.prompt = PromptTemplate(
#         input_variables=["history", "context", "question"],
#         template=st.session_state.template,
#     )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history", return_messages=True, input_key="question"
    )

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=vectorstore_directory,
        embedding_function=HuggingFaceEmbeddings(
            model_name=model_path, model_kwargs={"device": "cpu"}, # comma added
            encode_kwargs = {"normalize_embeddings": True} #added
        ),
    )

if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model=llmmodel,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")


uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if uploaded_file is not None:
    # # Create the complete file path
    file_path = os.path.join(file_directory, uploaded_file.name)
    if not os.path.isfile(file_path):
        with st.spinner("Saving your document..."):
            bytes_data = uploaded_file.read()
            with open(file_path, "wb") as f:
                f.write(bytes_data)

        with st.spinner("Analyzing your document..."):
            loader = PyPDFLoader(file_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                # 1500
                chunk_size=1024,
                chunk_overlap=100,
                length_function=len,
            )
            all_splits = text_splitter.split_documents(data)

            progress_bar = st.progress(0)
            for i, doc in enumerate(all_splits):
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=[doc],
                    embedding=HuggingFaceEmbeddings(model_name=model_path),
                    persist_directory=vectorstore_directory,
                )
                st.session_state.vectorstore.persist()
                progress_bar.progress((i + 1) / len(all_splits))

            progress_bar.empty()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                # "prompt": st.session_state.prompt,#prompt state
                "prompt": prompt,  # promtp RAG
                "memory": st.session_state.memory,
            },
        )

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
            st.markdown(response["result"])
        chatbot_message = {"role": "assistant", "message": response["result"]}
        st.session_state.chat_history.append(chatbot_message)
else:
    st.write("Please upload a PDF file.")

