from pdf_loader import load_pdf
from chunking import chunk_text
from embeddings import generate_embeddings, model
from vector_store import create_faiss_index
from retriever import search
from llm import generate_answer

documents = load_pdf(
    "data/2601.14123v1.pdf"
)

full_text = ""

for doc in documents:
    full_text += doc["text"] + "\n"

chunks = chunk_text(
    full_text,
    chunk_size=500,
    overlap=50
)

embeddings = generate_embeddings(chunks)

index = create_faiss_index(
    embeddings
)

question = input("Ask a question: ")

query_embedding = model.encode(question)

distances, indices = search(
    query_embedding,
    index,
    k=3
)

context = "\n\n".join(
    [chunks[idx] for idx in indices[0]]
)

answer = generate_answer(
    question,
    context
)

print("\nAnswer:")
print(answer)