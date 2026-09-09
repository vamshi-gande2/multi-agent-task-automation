import chromadb
from sentence_transformers import SentenceTransformer


DOCUMENTS = [
    {
        "id": "rag",
        "text": (
            "Retrieval-Augmented Generation (RAG) retrieves relevant "
            "documents and provides their context to a language model "
            "before generation."
        ),
    },
    {
        "id": "langgraph",
        "text": (
            "LangGraph is a framework for building stateful, graph-based "
            "agent workflows with nodes, edges, and conditional routing."
        ),
    },
    {
        "id": "multi_agent",
        "text": (
            "A multi-agent system uses specialized agents to divide and "
            "solve different parts of a task."
        ),
    },
]


# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Local Chroma database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="knowledge"
)


def build_index():
    """Create embeddings and store documents in Chroma."""
    embeddings = model.encode(
        [doc["text"] for doc in DOCUMENTS]
    ).tolist()

    collection.upsert(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        embeddings=embeddings,
    )


def retrieve(query: str, top_k: int = 2):
    """Retrieve the most semantically relevant documents."""
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    return results["documents"][0]


if __name__ == "__main__":
    build_index()

    query = input("Enter a question: ")

    results = retrieve(query)

    print("\nRetrieved context:")
    for result in results:
        print("-", result)
