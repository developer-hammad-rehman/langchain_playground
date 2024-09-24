# Import required types and modules
import os  # Module for interacting with the operating system
from dotenv import load_dotenv  # Function to load environment variables from a .env file
from langchain_postgres.vectorstores import PGVector  # Class for PostgreSQL vector storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Class for Google Generative AI embeddings

# Load environment variables from .env file
load_dotenv()

# Retrieve the PostgreSQL connection string from environment variables
connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")

# Modify the connection string to be compatible with the psycopg driver
connection_string = connection_string.replace(
    "postgresql", "postgresql+psycopg"
)

# Initialize the embeddings function using a model from Google Generative AI
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Specify the embedding model to use
)

# Create a vector store using PGVector with the specified parameters
vector_store = PGVector(
    connection=connection_string,            # Database connection string
    collection_name="document_collection",   # Name of the collection/table in the database
    embeddings=embedding,                    # Embedding function to encode texts
    use_jsonb=True,                          # Use JSONB data type for storing embeddings
)

# Create a retriever object from the vector store for similarity search
retrieval = vector_store.as_retriever(
    search_type="similarity",                # Define the type of search (similarity search)
    search_kwargs={"k": 10}                  # Number of similar documents to retrieve
)

# Invoke the retriever with a query to get similar documents
docs = retrieval.invoke("What Are GANS")  # Query string for retrieval

# Iterate over the retrieved documents and print their content
for doc in docs:
    print(doc.page_content, '\n\n')         # Print the content of each document
    print("__" * 80)                        # Print a separator line
