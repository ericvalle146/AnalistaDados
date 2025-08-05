from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL")


embeddings_model = OllamaEmbeddings(model="snowflake-arctic-embed2", base_url=OLLAMA_URL)


def ensure_chroma(path_pdf: str, persist_dir: str, chunk_size: int, chunk_overlap: int):

    if os.path.exists(persist_dir):
        print("Persistindo banco da existente...")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings_model)

    else:
        print("Criando banco vetorial, pode demorar alguns minutos...")
        loader = PyPDFLoader(path_pdf)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)

        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_dir
        )