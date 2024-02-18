import streamlit as st
import nltk
from nltk.corpus import stopwords
import networkx as nx
import helper


def read_articles(data):
    sentances = data.split(". ")
    output = []
    for sentance in sentances:
        output.append(sentance.replace("[^A-Za-z]", " ").split(" "))
    return output


def generate_summary(file_name, top_n=5):
    # nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences = read_articles(file_name)

    sentence_similarity_martix = helper.build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)),
                             reverse=True)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
        # summarize_text.append(" ")

    st.write("Summarize Text: \n\n", ". ".join(summarize_text))


st.header("Text Summarizer")

data = st.text_area(label="Enter the article here")
if st.button("Get Summary"):
    generate_summary(data, 3)
