import numpy as np
from nltk.cluster.util import cosine_distance


def sent_similarity(sent_1, sent_2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent_1 = [word.lower() for word in sent_1]
    sent_2 = [word.lower() for word in sent_2]

    all_words = list(set(sent_1+sent_2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent_1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent_2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sent_similarity(
                    sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix
