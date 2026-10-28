
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