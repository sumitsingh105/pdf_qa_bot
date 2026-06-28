from pdf_loader import load_pdf
from chunking import chunk_text
from embeddings import generate_embeddings

documents = load_pdf("data/2601.14123v1.pdf")

full_text = ""

for doc in documents:
    full_text += doc["text"] + "\n"

chunks = chunk_text(
    full_text,
    chunk_size=500,
    overlap=50
)

embeddings = generate_embeddings(chunks)

print("Total Chunks:", len(chunks))

print("Embedding Shape:", embeddings.shape)

print("\nFirst Embedding:")
print(embeddings[0][:10])