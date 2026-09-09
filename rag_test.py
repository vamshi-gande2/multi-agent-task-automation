from rag import retrieve, build_context

query = "What is RAG?"

results = retrieve(query)

assert len(results) > 0
assert results[0]["id"] == "rag"

context = build_context(query)

assert "Retrieval-Augmented Generation" in context

print("RAG test passed.")
print("\nRetrieved:")
for result in results:
    print("-", result["title"])
