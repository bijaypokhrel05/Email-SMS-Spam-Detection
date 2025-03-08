import streamlit as st
import pickle
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = [word for word in y if word not in stopwords.words('english') and word not in punctuation]

    # lematize our words
    stem_words = []
    for i in text:
        stem_words.append(ps.stem(i))

    return ' '.join(stem_words)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area("Enter the message")
# input_sms = input("Enter the message")

if st.button('Predict'):
    # --preprocess
    transformed_sms = text_transform(input_sms)
    # --vectorizer
    vector_input = tfidf.transform([transformed_sms])
    # --predict
    result = model.predict(vector_input)[0]
    # --display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')