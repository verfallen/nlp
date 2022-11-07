from cbow_neg_sample import CBOW as CBOW_NEG
from corpus import Corpus, load_data
import torch
import constant as C
import torch.nn.functional as F

text = list(load_data("../data/tonghua.txt"))
corpus = Corpus(text)
vocab = corpus.vocab

model_path = "./models/sg_neg32.pt"
model = torch.load(model_path)
model = model.to(torch.device("cpu"))


def predict(word):
    return model.pred(
        torch.LongTensor(corpus.encode(word)),
    )


def get_similar_words(word, k=10):
    target = predict(word)
    scores = []
    for i in range(len(vocab)):
        if vocab[i] == word:
            continue
        v = predict(vocab[i])
        cosine_sim = F.cosine_similarity(target, v).data.tolist()[0]
        scores.append([vocab[i], cosine_sim])
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]  # s


def analogy(words):
    a, b, c = words
    vec_a = predict(a)
    vec_b = predict(b)
    vec_c = predict(c)
    result = vec_a - vec_b + vec_c
    return get_similar_words(result, 5)


if __name__ == "__main__":
    words = input("input three word: ")
    words = tuple(words.split(" "))
    # print(analogy(words))

    a, b, c = words
    vec_a = predict(a)
    vec_b = predict(b)
    vec_c = predict(c)
    print(vec_a - vec_b + vec_c)
    # print(vocab[:1000])

    # word = input("input a word:")
    # print(get_similar_words(word))
