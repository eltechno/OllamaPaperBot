import re
import os
import glob
from io import BytesIO
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings




index_name="k8s-books"

def clean_text(text: str) -> str:
    """
    Cleans the extracted text from a PDF page.
    Args:
        text (str): The extracted text.
    Returns:
        str: The cleaned text.
    """
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    """
    Parses a PDF file and extracts cleaned text from each page.
    Args:
        file (BytesIO): The PDF file as a binary stream.
        filename (str): The name of the PDF file.
    Returns:
        Tuple[List[str], str]: A tuple containing a list of strings (one per page) and the filename.
    """
    try:
        pdf = PdfReader(file)
    except Exception as e:
        # Handle exceptions related to file reading
        raise ValueError(f"Error reading PDF file: {e}")

    output = []
    for page in pdf.pages:
        try:
            text = page.extract_text()
            if text:
                cleaned_text = clean_text(text)
                output.append(cleaned_text)
        except Exception as e:
            # Handle or log exceptions related to text extraction
            print(f"Error extracting text from page: {e}")

    return output, filename


def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            # chunk_size=4000,
            chunk_size=500,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i} #metadata hdp
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}" #aqui las paginas
            doc.metadata["filename"] = filename  # aqui el nombre de archivo
            doc_chunks.append(doc)
    return doc_chunks



########expe
def docs_to_index_experimental(docs, vectorstore_directory):
    #model_name = "BAAI/bge-large-en-v1.5"
    model_name = "hkunlp/instructor-large"
    model_kwargs = {"device": "mps"}
    encode_kwargs = {"normalize_embeddings": True}
    embed_instruction = "Represent the text from the Kubernetes documentation"
    query_instruction = "Query the most relevant text from the Kubernetes documentation"
    # Create embeddings using HuggingFaceEmbeddings with a specific Sentence Transformers model ##HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    embed_instruction=embed_instruction,
    query_instruction=query_instruction
    )
    # Create a FAISS vector store with the documents and embeddings
    index = FAISS.from_documents(docs, embeddings)
    # Save the FAISS vector store to the specified local path
    index.save_local(vectorstore_directory,index_name)
    return index
########Expe




def get_index_for_pdf(pdf_files, pdf_names, vectorstore_directory):
    documents = []
    for pdf_file, pdf_name in zip(pdf_files, pdf_names):
        text, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index_experimental(documents, vectorstore_directory)
    return index




def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def main():
    # Define the directory where your PDF files are stored
    pdf_directory = "./data"
    
    # Define the directory where the vector store will be saved
    vectorstore_directory = "./vectorstore/lab/"

    # List all PDF files in the directory
    pdf_paths = glob.glob(os.path.join(pdf_directory, "*.pdf"))

    # Read PDF files into memory and create corresponding names
    pdf_files = []
    pdf_names = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as file:
            pdf_files.append(file.read())
            pdf_names.append(os.path.basename(pdf_path))

    # Call the function with the PDF files, names, and vector store directory
    index = get_index_for_pdf(pdf_files, pdf_names, vectorstore_directory)

    # [Any additional code you want to execute after this point]

if __name__ == "__main__":
    main()

