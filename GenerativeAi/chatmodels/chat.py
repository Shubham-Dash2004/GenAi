# from dotenv import load_dotenv
# load_dotenv()  

# from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
# response = model.invoke("Hello, how are you?")
# print(response.content)

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import time

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

start = time.time()

response = model.invoke("Hello, how are you?")

print(response.content)
print(f"Time taken: {time.time() - start:.2f} seconds")