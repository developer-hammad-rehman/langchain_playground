# Import necessary libraries and modules
from typing import Literal
from chromadb import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the generative AI model using Google's API
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Set up the embeddings function using Google's embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# Define a function to query the vector database for relevant documents
def query_vector_db(
    store_name: str,
    query: str,
    embedding_func: Embeddings,
    search_type: Literal["similarity", "mmr", "similarity_score_threshold"],
    search_kwargs: dict,
):
    # Initialize Chroma vector store with specified embedding function and persistence directory
    vector_store = Chroma(
        embedding_function=embedding_func,
        persist_directory="vector_db/document_db",
        collection_name=store_name,
    )
    # Configure the retriever with the given search type and parameters
    retriever = vector_store.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    print(f"_____Searching Relevant Information from {store_name}_______")
    # Retrieve documents that are relevant to the query
    relevant_docs = retriever.invoke(query)
    # Concatenate the content of all retrieved documents into a single string
    response = "".join([doc.page_content for doc in relevant_docs])
    return response


# Prepare messages for interaction with the generative AI model
messages = [
    SystemMessage(content="You are the AI Text Summarizer"),
    HumanMessage(
        content=f""""Generate me the summary of document: \n 
                             Relevant document content: \n
                             {query_vector_db("document_db", "Generative AI", embedding, "similarity", {"k": 4})}
                             """
    ),
]

# Invoke the generative AI model with the prepared messages
response = model.invoke(messages)

# Print the output response from the model
print(response.content)
