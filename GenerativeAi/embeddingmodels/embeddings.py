
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    "Hello this is Akarsh Vyas",
    "Hello your name is YouTube",
    "And you all are very beautiful"
]

# Convert one query into a vector
vector = embeddings.embed_query("Hello, how are you?")

print("Vector dimensions:", len(vector))
print("First 10 numbers:", vector[:10])

# # Convert multiple documents into vectors
# vectors = embeddings.embed_documents(texts)

# print("Number of document vectors:", len(vectors))
# print("Dimensions per document:", len(vectors[0]))



# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# embeddings = OpenAIEmbeddings(
#     model = 'text-embedding-3-large',
#     dimensions=64
# )

# texts = [
#     "Hello this is Akarsh Vyas",
#     "Hello your name is YouTube",
#     "And you all are very beautiful"
# ]
# vector = embeddings.embed_query("Hello , how are you ?")
# # vector = embeddings.embed_documents(texts)

# print(vector)