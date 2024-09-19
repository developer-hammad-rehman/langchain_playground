from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda,  RunnableParallel 
from langchain.schema.output_parser import StrOutputParser
import json

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

base_message = [
    ("system" , "You are the professinal Content Writer"),
    ("human" , "Write me the linkdien post on {topic}")
]

challanges_message = [
    ("system" , "You have to figure the challanges of product"),
    ("human" , "{product_name}")
]

future_message = [
    ("system" , "You have to analyzie the future of the product"),
    ("human" , "{product_name}")
]

base_templeate = ChatPromptTemplate.from_messages(base_message)
challanges_tempelate = ChatPromptTemplate.from_messages(challanges_message)
future_templeate = ChatPromptTemplate.from_messages(future_message)

base_chain = base_templeate | model 

parrallel_chain = base_chain | RunnableParallel(branches={"challanges" : challanges_tempelate | model  , "future": future_templeate | model }) | RunnableLambda(lambda x : f'Challanges {x["branches"]["challanges"]} / Further {x["branches"]["future"]}') 

response  = parrallel_chain.invoke({"topic":"Generative AI"})

print(response)