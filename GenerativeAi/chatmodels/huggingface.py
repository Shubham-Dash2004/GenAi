
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    # task="text-generation",
    # max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Why do parrots talk?")
print(response.content)
