# Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model for use with Langchain
from langchain_core.messages import HumanMessage, SystemMessage  # Import message types used to communicate with the model
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables (e.g., API keys, configuration) from the .env file
load_dotenv()

# Initialize the Google Generative AI model with the specified model version 'gemini-1.5-flash'
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define a series of messages that form the conversation context
# SystemMessage provides context to the AI on how it should behave
# HumanMessage is the input or question that the human user asks
messages = [
    SystemMessage(content="You are the AI Researcher"),  # System role setup: defines AI's role in this conversation
    HumanMessage(content="What is the latest news of AI")  # Human input: a query to ask about the latest AI news
]

# Use the model to invoke a response based on the provided messages
response = model.invoke(messages)

# Output the AI's response content
print(response.content)


# --- Streaming With Messages ---

# In this section, we will handle the conversation in a streaming mode for chunked responses.

# Stream the response from the model in chunks (this is useful for large or ongoing outputs)
response = model.stream(messages)

# Iterate over each chunk of the streamed response
for chunk in response:
    # Print each chunk's content continuously, without a newline (to maintain a smooth, uninterrupted response flow)
    print(chunk.content, end="")
