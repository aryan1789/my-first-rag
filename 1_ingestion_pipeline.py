import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    print(f"Loading documents from: {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")
    
    loader = DirectoryLoader(path=docs_path, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in directory '{docs_path}'.")

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print(f"\nSplitting documents into chunks...")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    # if chunks:

    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\nChunk {i+1}:")
    #         print(f"Source: {chunk.metadata['source']}")
    #         print(f"Chunk content length: {len(chunk.page_content)} characters")
    #         print(f"Chunk content preview: ")
    #         print(f"{chunk.page_content}")  # Print first 100 characters
    #         print(f"-" * 50)

    #     if len(chunks) > 5:
    #         print(f"\n...and {len(chunks) - 5} more chunks.")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and vector store...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("---Creating vector store---")
    vectorestore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print("---Finished creating vector store---")

    print(f"Vector store created and savwed to {persist_directory}")
    return vectorestore

def main():
    print("Main function")

    #1 Loading the files
    documents = load_documents(docs_path="docs")

    #2 Splitting the documents into chunks
    chunks = split_documents(documents)
    #3 Creating the vector store and saving it to disk
    vector_store = create_vector_store(chunks)


if __name__ == "__main__":
    main()