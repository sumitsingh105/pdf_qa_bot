import numpy as np


def search(
    query_embedding,
    index,
    k=3
):

    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    return distances, indices