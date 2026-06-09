from dotenv import load_dotenv
load_dotenv()
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, download_embeddings, text_split
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY #type: ignore
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY #type: ignore

extracted_data = load_pdf_file(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(filter_data)

embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "intellicast"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embeddings,
    index_name=index_name
)