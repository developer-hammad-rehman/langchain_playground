# Import necessary modules and functions from various packages
from langchain.chains.retrieval import create_retrieval_chain  # Function to create a retrieval chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # Function to combine documents into a chain
from langchain_core.prompts import ChatPromptTemplate  # Class for creating chat prompt templates
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # Classes for Google Generative AI models
from langchain_postgres.vectorstores import PGVector  # Class for PostgreSQL vector storage
from dotenv import load_dotenv  # Function to load environment variables from a .env file
import os  # Module for interacting with the operating system

# Load environment variables from a .env file into the script's environment
load_dotenv()

# Retrieve the PostgreSQL connection string from environment variables
connection_string = os.getenv("PGVECTOR_CONNECTION_STRING")

# Modify the connection string to be compatible with the psycopg driver
connection_string = connection_string.replace(
    "postgresql", "postgresql+psycopg"
)

# Initialize the embedding function using Google Generative AI embeddings
embedding_func = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"  # Specify the embedding model to use
)

# Initialize the language model (LLM) using Google's Generative AI chat model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # Specify the LLM model to use

# Set up the vector store (database) for storing document embeddings
vector_store = PGVector(
    connection=connection_string,  # Database connection string
    collection_name="document_collection",  # Name of the collection/table in the database
    embeddings=embedding_func,  # Embedding function to encode documents
    use_jsonb=True  # Use JSONB data type for storing embeddings
)

# Create a retriever object from the vector store for similarity search
retrieval = vector_store.as_retriever(
    search_type="similarity",  # Define the type of search (similarity search)
    search_kwargs={"k": 5}  # Number of similar documents to retrieve
)

# Create a chat prompt template with predefined system and human messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You have to provide the summary of the given {context}"),  # System message with a context placeholder
    ("human", "{input}")  # Human message with an input placeholder
])

# Create a chain that processes documents using the LLM and the prompt template
llm_chain = create_stuff_documents_chain(
    llm=llm,  # Language model to use
    prompt=prompt_template  # Prompt template for the conversation
)

# Combine the retriever and the LLM chain into a retrieval chain
retrieval_chain = create_retrieval_chain(
    retrieval,  # Retriever object for fetching documents
    llm_chain  # Chain to process the retrieved documents
)

# Invoke the retrieval chain with a user input to generate a summary
result = retrieval_chain.invoke({"input": "Generate me the summary of the book"})

# Print the generated summary from the result
print(result["answer"])
