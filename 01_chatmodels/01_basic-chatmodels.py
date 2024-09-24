# Import necessary modules
from dotenv import load_dotenv  # To load environment variables from a .env file
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model


# Load environment variables from a .env file (this might include API keys, etc.)
load_dotenv()

# Initialize the Google Generative AI model
# Using the model name 'gemini-1.5-flash', assuming this is the version required
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Query the model to ask a philosophical question
response = model.invoke("What is the meaning of life?")

# Print The Response

print(response.content)