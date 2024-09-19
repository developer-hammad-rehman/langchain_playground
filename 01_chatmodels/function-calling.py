from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

# Load environment variables from the .env file (this could contain API keys, model credentials, etc.)
load_dotenv()

# Initialize the ChatGoogleGenerativeAI model with the gemini-1.5-flash version
# This model will be used to generate conversational responses
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define the initial user message as a HumanMessage object
# This simulates the user asking about "Sir Zia"
messages = [
    HumanMessage("Tell Me About Sir Zia")
]

# Define a function that returns information about a teacher based on their name
def get_teacher_info(name: str):
    """
    This function takes a teacher's name as input and returns information about the teacher.
    It utilizes a dictionary to store data about various teachers for quick lookup.

    :param name: The name of the teacher to look up.
    :return: A dictionary containing details about the teacher, such as name, role, skills, and biography.
    """
    
    # A dictionary containing information about different teachers
    teachers = {
        "ammen alam": {
            "name": "Ammen Alam",
            "role": "Lead Teacher",
            "master_skill": "DevOps Master",
            "bio": "Ammen Alam has 10+ years of experience in the industry, specializing in DevOps and cloud solutions."
        },
        "sir qasim": {
            "name": "Muhammad Qasim",
            "role": "Generative AI Lead Teacher",
            "master_skill": "Machine Learning & AI",
            "bio": "Muhammad Qasim is a thought leader in AI and machine learning, currently guiding students in generative AI."
        },
        "sir zia": {
            "name": "Zia Khan",
            "role": "CEO of PIAIC, Panacloud, Panaversity, and Panaverse",
            "master_skill": "Cloud Applied Generative AI",
            "bio": "Zia Khan is the visionary behind Panacloud and a leader in applied generative AI, with extensive industry experience."
        }
    }

    # Normalize the input name by converting it to lowercase and removing any surrounding whitespace
    # This helps to ensure the function works even if the input has case differences or extra spaces
    name = name.lower().strip()
    
    # Return the teacher's information if the name exists in the dictionary; otherwise, return an error message
    if name in teachers:
        return teachers[name]
    else:
        return {"error": "Teacher not found. Please check the name and try again."}

# Define a Pydantic model for handling teacher information retrieval
class GetTeacherInfo(BaseModel):
    """Get The Information of Teachers"""
    name : str = Field(description="The Name of The Teacher")



# Create a list of tools (functions) to be used with the model
# Here, the model will be able to call the GetTeacherInfo tool when it needs to retrieve teacher info
tools = [GetTeacherInfo]

# Bind the tools to the AI model, allowing the model to use them during conversation
llm_with_tool = model.bind_tools(tools)

# Invoke the AI model with the initial user message, allowing the model to generate a response
ai_message = llm_with_tool.invoke(messages)

# Append the AI's response to the conversation history (messages list)
messages.append(ai_message)

# Define a dictionary that maps tool names (as called by the AI) to their corresponding Python functions
# This will allow the AI to call the get_teacher_info function when it identifies a tool call for teacher info
avaliable_func = {
    "GetTeacherInfo": get_teacher_info
}

# Check if the AI has called any tools (e.g., requesting teacher info)
if ai_message.tool_calls:
    for tool_call in ai_message.tool_calls:
        # Extract the function/tool name and arguments from the tool call
        funtion_name = tool_call.get("name")
        func_args = tool_call.get("args")
        tool_call_id = tool_call.get("id")

        # Map the tool name to the corresponding Python function
        function_to_call = avaliable_func[funtion_name]

        # Call the function with the provided arguments, spreading the dictionary keys as keyword arguments
        function_output = function_to_call(**func_args)

        # Append the function's output to the conversation history as a ToolMessage
        messages.append(ToolMessage(content=function_output, tool_call_id=tool_call_id))
    
    # Invoke the AI model again, providing it with the updated conversation history (including the tool's output)
    response = llm_with_tool.invoke(messages)
    
    # Print the final response from the AI to the console
    print(response.content)
else:
    # If no tools were called, simply print the AI's generated response
    print(ai_message.content)