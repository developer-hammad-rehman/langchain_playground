# Import necessary modules from LangChain and Google Generative AI libraries
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


# Load environment variables from .env file
load_dotenv()


# Initialize a PDF loader to load a document from a specified path
loader = PyPDFLoader(
    "./document/genai.pdf"  # Path to the PDF document
)

# Set up the embeddings function using a specified model from Google Generative AI
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Load the document content from the PDF file
document = loader.load()

# Initialize a text splitter to divide the document into manageable chunks
# Each chunk will have up to 1000 characters with no overlap between chunks
text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Split the loaded document into smaller chunks according to the defined splitter settings
docs = text_spliter.split_documents(document)

# Create a vector store and populate it with the document chunks
# Each chunk is converted into embeddings and then stored
Chroma.from_documents(
    docs,  # List of document chunks to be processed
    embedding=embedding,  # Embedding function to convert text into vectors
    persist_directory="vector_db/document_db",  # Directory for storing the vector data
    collection_name="document_db"  # Name of the collection within the vector store
)
