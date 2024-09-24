# Import necessary modules from various packages
from dotenv import load_dotenv  # Function to load environment variables from a .env file
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Class for Google Generative AI embeddings
from langchain_postgres.vectorstores import PGVector # Class for PostgreSQL vector store
import os  # Module for interacting with the operating system

# Load environment variables from the .env file into the script's environment
load_dotenv()

# Retrieve the PostgreSQL connection string from environment variables
connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")

# Modify the connection string to be compatible with the psycopg driver
connection_string = connection_string.replace(
    "postgresql", "postgresql+psycopg"
)

# Define a sample text to be processed
text = "Langchain is the powerful library for large language models"

# Initialize the embedding function using a specified model from Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Specify the embedding model to use
)

# Create a vector store using the PGVector class.
vector_store = PGVector(
    embeddings=embeddings,  # Embedding function to encode texts
    collection_name="text_db",  # Name of the collection/table in the database
    connection=connection_string,  # Database connection string
    use_jsonb=True,  # Use JSONB data type for storing embeddings
)

# Add the text to the vector store by converting it into embeddings
vector_store.add_texts([text])  # Add the text to the vector store
