from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

response = llm.invoke("Create a full D&D adventure with setting, plot hooks, encounters, and NPCs.")

print(response.content)