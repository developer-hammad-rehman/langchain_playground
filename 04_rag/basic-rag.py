# Import necessary modules from langchain_google_genai and langchain_community
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

# Define a sample text to be processed
text = "Langchain is the powerful library for large language models"

# Initialize the embeddings function using a specified model from Google Generative AI
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# Create a vector store using the Chroma class. This involves:
# 1. Converting the provided text into embeddings using the initialized embedding model.
# 2. Storing these embeddings in a specified collection and directory.
vector_store = Chroma.from_texts(
    [text],  # The list of texts to embed; here it's just one text.
    embedding=embeddings,  # The embedding function to use.
    collection_name="text_db",  # The name of the collection where the vectors will be stored.
    persist_directory="vector_db/text_db"  # The directory path where the data will be persistently stored.
)


