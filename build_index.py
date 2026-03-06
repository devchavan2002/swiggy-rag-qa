from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading PDF...")

loader = PyPDFLoader("data/swiggy_report.pdf")
documents = loader.load()

print("Splitting text...")

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", ".", " "]
# )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " "]
)

chunks = splitter.split_documents(documents)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")

print("Vector index created successfully!")