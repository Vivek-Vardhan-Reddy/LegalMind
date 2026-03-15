def retrieve(query, embedder, store):

    q_vec = embedder.embed(query)

    chunks = store.search(q_vec, k=3)

    return chunks