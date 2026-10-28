import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(page_title="RAG Book Assistant")

st.title("📚 RAG Book Assistant")
st.write("Upload a PDF and ask questions from the document")

uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    
if st.button("Create Vector Database"):
    with st.spinner("Processing document..."):
        try:
            # Save the uploaded PDF temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name

            try:
                # 1. Load PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                st.write("Pages loaded:", len(docs))

                # 2. Split text into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(docs)

                # Remove empty chunks
                chunks = [
                    chunk for chunk in chunks
                    if chunk.page_content.strip()
                ]

                st.write("Text chunks created:", len(chunks))

                # Stop if PDF text extraction failed
                if not chunks:
                    st.error(
                        "No readable text was extracted from this PDF. "
                        "Check whether it is scanned or image-based."
                    )
                    st.stop()

                # 3. Initialize embedding model
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Test the embedding model
                test_vector = embeddings.embed_query("Test query")
                st.write("Embedding dimensions:", len(test_vector))

                if not test_vector:
                    st.error("The embedding model returned an empty vector.")
                    st.stop()

                # 4. Create and save Chroma database
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

                st.success(
                    f"Vector database created with {len(chunks)} chunks!"
                )

            finally:
                os.unlink(file_path)

        except Exception as e:
            st.error(f"Database creation failed: {e}")




if os.path.exists("chroma_db") and os.listdir("chroma_db"):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.divider()
    st.subheader("Ask Questions From the Book")

    query = st.text_input("Enter your question")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        st.write("### AI Answer")
        st.write(response.content)