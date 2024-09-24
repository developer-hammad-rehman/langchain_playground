# Import the ChatPromptTemplate class from langchain_core to help structure and format prompts
from langchain_core.prompts import ChatPromptTemplate

# Example 1: Creating a simple template prompt

# Create a ChatPromptTemplate using the `from_template` method.
# The template expects a variable named "topic" to be provided when invoked.
prompt = ChatPromptTemplate.from_template("What is {topic}")

# Call the `invoke` method with the variable "topic" set to "AI".
# This fills in the placeholder {topic} with the value "AI" and creates the prompt: "What is AI".
print(prompt.invoke({"topic": "AI"}))


# Example 2: Creating a multi-message template

# Define a series of messages using tuples.
# Each message consists of two parts:
# - The message type ("system" or "human") indicating whether it's a system message or user input.
# - The message template with placeholders for variables (e.g., {programing_language}, {app_name}).
messages = [
    ("system", "You are the senior developer in {programing_language}"),  # System message setting up the role
    ("human", "Write me the code for {app_name}")  # Human (user) message asking for code related to an app
]

# Create a ChatPromptTemplate using the `from_messages` method.
# This allows multiple message templates to be created at once with placeholders for different variables.
prompt = ChatPromptTemplate.from_messages(messages)

# Call the `invoke` method with two variables:
# - "programing_language" set to "Python"
# - "app_name" set to "Todo List"
# The system message becomes: "You are the senior developer in Python."
# The human message becomes: "Write me the code for Todo List."
print(prompt.invoke({"programing_language": "Python", "app_name": "Todo List"}))
