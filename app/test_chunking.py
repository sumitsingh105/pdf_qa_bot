from pdf_loader import load_pdf
from chunking import chunk_text

# Load PDF
documents = load_pdf("data/2601.14123v1.pdf")

# Combine all pages into one text
full_text = ""

for doc in documents:
    full_text += doc["text"] + "\n"

# Save extracted text for inspection
with open("extracted_text.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print("Extracted text saved to extracted_text.txt")

# Create chunks
chunks = chunk_text(
    text=full_text,
    chunk_size=500,
    overlap=50
)

# Add metadata
all_chunks = []

for idx, chunk in enumerate(chunks):

    all_chunks.append({
        "chunk_id": idx + 1,
        "text": chunk
    })

# Statistics
print("\n" + "=" * 50)
print(f"Total Pages: {len(documents)}")
print(f"Total Chunks: {len(all_chunks)}")
print("=" * 50)

# Print first 3 chunks
print(all_chunks[:3])