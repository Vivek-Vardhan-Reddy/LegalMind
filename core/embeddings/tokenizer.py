class Tokenizer:
    def __init__(self):
        self.word2idx={"<PAD>":0,"<UNK>":1}
        self.idx2word={0:"<PAD>",1:"<UNK>"}
        self.vocab_size=2


    def build_vocab(self,texts):
        for text in texts:
            for word in text.lower().split():
                if word not in self.word2idx:
                    self.word2idx[word]=self.vocab_size
                    self.idx2word[self.vocab_size]=word
                    self.vocab_size+=1


    def encode(self,text,m,max_len=20):
        tokens=text.lower().split()
        ids=[self.word2idx.get(w,1) for w in tokens]
        ids=ids[:max_len]
        ids+=[0]*(max_len-len(ids))

        return ids