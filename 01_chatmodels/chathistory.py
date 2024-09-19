# Import necessary modules
from langchain_google_genai import ChatGoogleGenerativeAI  # Import the Google Generative AI model for interaction
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # Import message types for system, human, and AI communication
from dotenv import load_dotenv  # Import dotenv to load environment variables from .env file

# Load environment variables (e.g., API keys, configuration settings) from the .env file
load_dotenv()

# Initialize chat history as an empty list to store a sequence of messages (System, Human, and AI messages)
chathistory: list[SystemMessage, AIMessage, HumanMessage] = []

# Initialize the ChatGoogleGenerativeAI model with the specified version 'gemini-1.5-flash'
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Create the system message to set the context for the AI
# SystemMessage is used to define the role or behavior of the AI in the conversation
system_message: SystemMessage = SystemMessage("You are a helpful AI Assistant")

# Add the system message to the chat history to inform the model of its role
chathistory.append(system_message)

# Start an interactive loop where the user can input messages and get responses from the AI
while True:
    # Get the user's message (prompt) from the console input
    prompt = input("Write Your Message (or 'e' to exit): ")
    
    # Check if the user wants to exit the loop by typing 'e'
    if prompt.lower() == "e":
        break  # Exit the loop when 'e' is entered by the user
    
    # If not exiting, create a HumanMessage from the user's input and add it to the chat history
    else:
        prompt = HumanMessage(content=prompt)
        chathistory.append(prompt)
        
        # Invoke the model with the chat history (includes system, human, and previous AI messages)
        response = model.invoke(chathistory)
        
        # Print the AI's response to the console
        print("AI: ", response.content)
        
        # Add the AI's response as an AIMessage to the chat history for context in future interactions
        chathistory.append(response)

# Once the loop is exited, print the entire chat history (for debugging or review purposes)
print(chathistory)
