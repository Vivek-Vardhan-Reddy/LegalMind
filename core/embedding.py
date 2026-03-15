import math
from collections import Counter


class Tokenizer:

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, texts):

        for text in texts:

            for word in text.lower().split():

                if word not in self.word2idx:

                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word

                    self.vocab_size += 1

    def tokenize(self, text):

        return text.lower().split()


def positional_encoding(seq_len, d_model):
    """
    Generate positional encoding matrix
    """

    pe = [[0] * d_model for _ in range(seq_len)]

    for pos in range(seq_len):

        for i in range(0, d_model, 2):

            pe[pos][i] = math.sin(pos / (10000 ** (i / d_model)))

            if i + 1 < d_model:
                pe[pos][i + 1] = math.cos(pos / (10000 ** (i / d_model)))

    return pe


class EmbeddingModel:

    def __init__(self):

        self.tokenizer = Tokenizer()
        self.idf = {}

    def fit(self, documents):

        # build vocabulary
        self.tokenizer.build_vocab(documents)

        N = len(documents)

        df = Counter()

        for doc in documents:

            words = set(self.tokenizer.tokenize(doc))

            for w in words:
                df[w] += 1

        for word in self.tokenizer.word2idx:

            if word in ["<PAD>", "<UNK>"]:
                continue

            self.idf[word] = math.log((N + 1) / (df[word] + 1)) + 1

    def embed(self, text):

        tokens = self.tokenizer.tokenize(text)

        seq_len = len(tokens)

        vector = [0] * self.tokenizer.vocab_size

        tf = Counter(tokens)

        # TF-IDF embedding
        for word, count in tf.items():

            idx = self.tokenizer.word2idx.get(word, 1)

            idf_val = self.idf.get(word, 1)

            vector[idx] = count * idf_val

        # Positional Encoding
        pe = positional_encoding(seq_len, self.tokenizer.vocab_size)

        for pos in range(seq_len):

            for i in range(min(len(vector), len(pe[pos]))):

                vector[i] += pe[pos][i]

        return vector