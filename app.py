from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings
from src.prompt import *

import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

embeddings = download_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="intellicast",
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY, #type: ignore
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")

        response = rag_chain.invoke({
            "input": msg
        })

        return response["answer"]

    except Exception as e:
        print("ERROR:", e)
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)