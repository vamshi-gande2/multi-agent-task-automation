"""
Simple local RAG retrieval module.

This MVP uses a small document collection and keyword-based retrieval.
It is intentionally lightweight so the retrieval flow can be tested
without external API keys or model downloads.
"""

DOCUMENTS = [
    {
        "id": "rag",
        "title": "Retrieval-Augmented Generation",
        "content": (
            "Retrieval-Augmented Generation (RAG) retrieves relevant "
            "documents or passages and provides that context to a language "
            "model before generation. This helps ground responses in a "
            "specific knowledge source."
        ),
    },
    {
        "id": "langgraph",
        "title": "LangGraph",
        "content": (
            "LangGraph is a framework for building stateful agent workflows "
            "using graph-based nodes, edges, and conditional routing."
        ),
    },
    {
        "id": "agents",
        "title": "Multi-Agent Systems",
        "content": (
            "A multi-agent system uses specialized agents that can perform "
            "different tasks and collaborate or route work between them."
        ),
    },
]


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Return the most relevant local documents for a query."""
    query_words = {
        word.lower().strip(".,!?():;")
        for word in query.split()
        if len(word) > 2
    }

    scored_documents = []

    for document in DOCUMENTS:
        text = (
            document["title"] + " " + document["content"]
        ).lower()

        score = sum(
            1 for word in query_words
            if word in text
        )

        if score > 0:
            scored_documents.append((score, document))

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored_documents[:top_k]
    ]


def build_context(query: str, top_k: int = 2) -> str:
    """Build context from retrieved documents."""
    documents = retrieve(query, top_k)

    if not documents:
        return "No relevant documents were retrieved."

    return "\n\n".join(
        f"[{doc['title']}]\n{doc['content']}"
        for doc in documents
    )


if __name__ == "__main__":
    query = input("Enter a question: ")

    results = retrieve(query)

    print("\nRetrieved Documents:")
    for result in results:
        print(f"- {result['title']}")

    print("\nContext:")
    print(build_context(query))
