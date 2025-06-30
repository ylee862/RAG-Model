## Class for storing a piece of text and assiciated metadata
from langchain.schema import Document

## To load all documents
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

## To split the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

## To store the documents
from langchain_chroma import Chroma

## Use HuggingFace for Embedding
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import os

## Offers high-level operations on files and collections of files
import shutil

DATA_PATH = "data/articles/article"
CHROMA_PATH = "chroma"

## Main function
def main():
    generate_data_store()
    

## To generate indexing
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    

def load_documents():
    
    ## Using DirectoryLoader class to retrieve loader object
    ## Loading '.md' documents in DATA_PATH into loader variable
    loader = PyPDFDirectoryLoader(DATA_PATH)
    
    ## Loading the data
    documents = loader.load()
    
    return documents


## Text splitters breaking large documnets into smaller chunks
def split_text(documents: list[Document]): 
    text_splitter = RecursiveCharacterTextSplitter(
    
    ## Chunk size about 1000 characters
    chunk_size = 1000,
    
    ## Each chunks are going to have an overlap of 500 characters
    chunk_overlap = 500,
    length_function = len,
    add_start_index = True,
    is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    
    ## Printing out the number of original documents and number of chunks it was split into
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    
    ## Picked out a random chunk and printed out the data
    document = chunks[40]
    print(document.page_content)
    print(document.metadata)
    
    return chunks

## To query each chunk, we must turn it into a database, storing our splits
def save_to_chroma(chunks: list[Document]):
    
    ## Clear out the database first if it already exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        
    ## Create a new DB from the documents
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        
        ## Persist Directory stores the database on the system
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == '__main__':
    main()
    
