# Import required types and modules
from typing import Literal
from chromadb import Embeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize the embeddings function using a model from Google Generative AI
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Define a function to query a vector database and retrieve documents based on search criteria
def query_vector_db(
        store_name: str,  # Name of the vector store
        query: str,  # Query string to search for relevant documents
        embedding_func: Embeddings,  # Embedding function to generate document vectors
        search_type: Literal["similarity", "mmr", "similarity_score_threshold"],  # Type of search method to use
        search_kwargs: dict  # Additional parameters specific to the search method
):
    # Initialize a Chroma vector store with the specified settings
    vector_store = Chroma(
        embedding_function=embedding_func,
        persist_directory="vector_db/document_db",  # Directory for storing vectors
        collection_name="document_db"  # Collection within the store for organizing data
    )
    
    # Configure the vector store to use a specific retrieval method based on the search type
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    # Log the start of a search operation
    print(f"_____Searching Relevant Information from {store_name}_______")
    
    # Perform the document retrieval
    relevant_docs = retriever.invoke(query)
    
    # Print the content of each relevant document found
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Doc : {i} // {doc.page_content}", "\n\n")  # Format the output for clarity

# Example query to find the author
query = "Who is the Author ?"

# Perform queries using different search types
query_vector_db("document_db", query, embedding, "similarity", {"k":3})  # Using similarity search
query_vector_db("document_db", query, embedding, "mmr", {"k": 3})  # Using max marginal relevance (MMR)
query_vector_db("document_db", query, embedding, "similarity_score_threshold", {"k": 3 , "score_threshold":0.1})  # Using Similarity Score Threshold