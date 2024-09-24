# Import necessary modules from LangChain and Google Generative AI libraries
import os  # Module for interacting with the operating system
from dotenv import load_dotenv  # Function to load environment variables from a .env file
from langchain_community.document_loaders import PyPDFLoader  # Class to load PDF documents
from langchain_postgres.vectorstores import PGVector  # Class for PostgreSQL vector storage
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Class for Google Generative AI embeddings
from langchain.text_splitter import CharacterTextSplitter  # Class to split text into chunks

# Load environment variables from .env file
load_dotenv()

# Retrieve the PostgreSQL connection string from environment variables
connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")

# Modify the connection string to be compatible with the psycopg driver
connection_string = connection_string.replace(
    "postgresql", "postgresql+psycopg"
)

# Initialize a PDF loader to load a document from a specified path
loader = PyPDFLoader(
    "./document/genai.pdf"  # Path to the PDF document
)

# Set up the embeddings function using a specified model from Google Generative AI
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Specify the embedding model to use
)

# Load the document content from the PDF file
document = loader.load()

# Initialize a text splitter to divide the document into manageable chunks
# Each chunk will have up to 1000 characters with no overlap between chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # Maximum size of each text chunk
    chunk_overlap=0   # No overlap between chunks
)

# Split the loaded document into smaller chunks according to the defined splitter settings
docs = text_splitter.split_documents(document)

# Create a vector store using PGVector and populate it with the document chunks
vector_store = PGVector(
    embeddings=embedding,      # Embedding function to encode texts
    collection_name="document_collection",     # Name of the collection/table in the database
    connection=connection_string,  # Database connection string
    use_jsonb=True                     # Use JSONB data type for storing embeddings
)

# Add the document chunks to the vector store by converting them into embeddings
vector_store.add_documents(docs)
