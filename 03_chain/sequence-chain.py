from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

capital_country_prompt = ChatPromptTemplate.from_template("What is the capital of the {country}")

famous_food_prompt = ChatPromptTemplate.from_template("Suggest me the place to visit {city}")

country_chain = capital_country_prompt | model 

food_chain = famous_food_prompt | model 



chain = country_chain | food_chain

for chunk in chain.stream({"country":"Pakistan"}):
    print(chunk.content , end="")