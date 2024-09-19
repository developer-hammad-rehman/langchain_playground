# Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model for use in Langchain
from dotenv import load_dotenv  # To load environment variables from a .env file

# Load environment variables (e.g., API keys, model credentials) from the .env file
load_dotenv()

# Initialize the Google Generative AI model with the specified model version 'gemini-1.5-flash'
# This object will be used to interact with the model for streaming responses
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Invoke the model to start a streaming conversation with the question "What is the meaning of life?"
# The stream method allows for chunked responses, which may be useful for large or progressive responses
response = model.stream("What is the meaning of life?")

# Iterate over each chunk of the streamed response
for chunk in response:
    # Print each chunk's content as it comes in, without adding a newline after each chunk
    # 'end=""' ensures that chunks are printed continuously on the same line
    print(chunk.content, end="")