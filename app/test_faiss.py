from pdf_loader import load_pdf
from chunking import chunk_text
from embeddings import generate_embeddings
from embeddings import model

from vector_store import create_faiss_index
from retriever import search


documents = load_pdf(
    "data/2601.14123v1.pdf"
)

full_text = ""

for doc in documents:
    full_text += doc["text"] + "\n"


raw_chunks = chunk_text(
    full_text,
    chunk_size=500,
    overlap=50
)

all_chunks = []

for idx, chunk in enumerate(raw_chunks):
    all_chunks.append({
        "chunk_id": idx,
        "text": chunk
    })

embeddings = generate_embeddings(
    [chunk["text"] for chunk in all_chunks]
)

index = create_faiss_index(
    embeddings
)

query = "What does the paper say about overlap?"

query_embedding = model.encode(query)

distances, indices = search(
    query_embedding,
    index,
    k=3
)

print("\nDistances:")
print(distances)

print("\nIndices:")
print(indices)

print("\nTop Matches")

for idx in indices[0]:

    print("\n" + "=" * 50)
    print(f"Chunk ID: {all_chunks[idx]['chunk_id']}")
    print("=" * 50)

    print(all_chunks[idx]["text"])