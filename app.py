import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

# Load embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)


# Load vector database

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# Retriever

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k":5,
        "score_threshold":0.4
    }
)


# Load Local LLM

llm = Ollama(model="mistral")



# Generate Answer

def generate_answer(query):

    # -- Debug similarity scores --
    results = vectorstore.similarity_search_with_score(query, k=5)

    with st.sidebar:
        st.subheader("Similarity Debug")
        for doc, score in results:
            st.write(f"Page {doc.metadata['page']+1} → {score:.3f}")

    # -- Query Expansion --
    expanded_query = (
        query +
        " Instamart services quick commerce grocery delivery "
        "dark stores product offerings Swiggy Instamart"
    )

    docs = retriever.invoke(expanded_query)

    # -- Filter irrelevant sections --
    filtered_docs = []
    for doc in docs:
        text = doc.page_content.lower()

        if "director" in text or "auditor" in text or "board" in text:
            continue

        filtered_docs.append(doc)

    docs = filtered_docs

    # -- If nothing retrieved 
    if not docs:
        return "The answer is not available in the document.", [], []

    # - Build context 
    context = "\n\n".join([doc.page_content for doc in docs])

    pages = sorted(list(set([doc.metadata["page"] + 1 for doc in docs])))

    # - Prompt 
    prompt = f"""
You are an AI assistant answering questions about the Swiggy Annual Report.

STRICT RULES:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not present in the context say:
   "The answer is not available in the document."
4. Always answer clearly and concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, pages, docs



# Streamlit UI

st.title("Swiggy Annual Report RAG QA")

query = st.text_input("Ask a question about the report:")

if query:

    answer, pages, docs = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources (Page Numbers)")
    st.write(", ".join([f"Page {p}" for p in pages]))

    st.subheader("Retrieved Context")

    for doc in docs:
        st.write(f"Page {doc.metadata['page'] + 1}")
        st.write(doc.page_content[:300] + "...")
        st.write("---")
