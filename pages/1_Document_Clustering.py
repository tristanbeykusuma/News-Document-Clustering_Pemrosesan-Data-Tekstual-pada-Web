import streamlit as st
import pickle
import numpy as np

loaded_kmeans = pickle.load(open('./model/clustering_model.sav', 'rb'))

vectorizer = pickle.load(open("./model/tfidf_vectorizer.pkl", "rb"))

clusters = ['sport', 'technology', 'religion']

def preprocess_text(text):
    text_vectorized = vectorizer.transform(text)
    return text_vectorized

def predict_cluster(text):
    text_vectorized = preprocess_text(text)
    predictions = loaded_kmeans.predict(text_vectorized)
    return clusters[predictions[0]], predictions

def main():
    st.title("News Document Clustering")

    uploaded_file = st.file_uploader("Upload a text document...", type="txt")
    if uploaded_file is not None:
        data = uploaded_file.read()
        st.write(data)
        data = [data]
        cluster, predictions = predict_cluster(data)
        st.write("Cluster:", predictions[0])
        st.write("Cluster Type:", cluster)

if __name__ == '__main__':
    main()
