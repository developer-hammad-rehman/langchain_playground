# Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model for chat interaction
from langchain_community.chat_message_histories import PostgresChatMessageHistory  # Import Postgres-based message history
from dotenv import load_dotenv  # To load environment variables from a .env file
import os  # Import os to access environment variables

# Load environment variables (e.g., database URL, API keys) from the .env file
load_dotenv()

# Retrieve the database connection string from environment variables
# Ensure that the environment variable "DATA_BASE_URL" exists in your .env file
connection_string = os.getenv("DATA_BASE_URL")

# Create a unique session ID for the user
# This ensures that each session's chat history is stored separately in the database
session_id = "user_session_id"

# Initialize a PostgresChatMessageHistory object to store chat messages in a PostgreSQL database
# This will enable persistent storage of the conversation history for the specified session
history = PostgresChatMessageHistory(
    connection_string=connection_string,  # The connection string to the Postgres database
    session_id=session_id  # The unique session ID for this conversation
)

# Display existing messages for this session (if any) from the database
print("____Messages____")
print(history.messages)  # Prints any existing messages stored in this session's history

# Initialize the Google Generative AI model with the version 'gemini-1.5-flash'
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

print("___START CONVERSATION_____")

# Begin an infinite loop to allow the user to continuously input messages and receive AI responses
while True:
    # Prompt the user for input
    user_message = input("Write Your Message (or 'e' to exit): ")
    
    # Add the user's input to the message history
    history.add_user_message(user_message)
    
    # Invoke the model with the current message history to generate a response
    # The history.messages provides the full context of previous messages (both user and AI)
    response = model.invoke(history.messages)
    
    # Print the AI's response to the console
    print("AI: ", response.content)
    
    # Add the AI's response to the message history to keep track of the conversation context
    history.add_ai_message(response.content)

# Note: The loop will continue running until the user interrupts it manually or adds an exit condition.
