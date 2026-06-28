from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

sentences = [
    "What is chunking?",
    "Explain chunking strategy",
    "How to cook pasta?"
]

embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(
    embeddings,
    embeddings
)

print(similarity_matrix)