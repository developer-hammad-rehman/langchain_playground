from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser


load_dotenv()


messages = [
    ("system" , "You are the senior developer of {language}"),
    ("human" , "{input}")
]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


prompt_tempelate = ChatPromptTemplate.from_messages(messages)


chain = prompt_tempelate | model | StrOutputParser()


result = chain.invoke({"language":"Python" , "input" : "How to do hello world"})

print(result)