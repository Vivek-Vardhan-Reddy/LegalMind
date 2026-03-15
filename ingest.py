from core.chunker import TextChunker
from core.embedding import EmbeddingModel
from core.vector_store import VectorStore


def load_document(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_index(path):

    # load contract
    text = load_document(path)

    # chunk the document
    chunker = TextChunker()
    chunks = chunker.chunk(text)

    # build embeddings
    embedder = EmbeddingModel()
    embedder.fit(chunks)

    # vector store
    store = VectorStore()

    for chunk in chunks:
        vector = embedder.embed(chunk)
        store.add(chunk, vector)

    return embedder, store