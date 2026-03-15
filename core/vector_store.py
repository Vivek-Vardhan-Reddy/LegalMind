import math


class VectorStore:

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, text, vector):

        self.texts.append(text)
        self.vectors.append(vector)

    def cosine_similarity(self, v1, v2):

        dot = sum(a*b for a,b in zip(v1,v2))

        norm1 = math.sqrt(sum(a*a for a in v1))
        norm2 = math.sqrt(sum(b*b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot/(norm1*norm2)

    def search(self, query_vector, k=3):

        scores = []

        for i,vec in enumerate(self.vectors):

            sim = self.cosine_similarity(query_vector, vec)

            scores.append((sim,i))

        scores.sort(reverse=True)

        results = []

        for score,index in scores[:k]:

            results.append(self.texts[index])

        return results