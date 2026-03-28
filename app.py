import streamlit as st
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Fake News Detection App")

user_input = st.text_area("Enter News Text:")

if st.button("Check"):
    if user_input:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)

        if prediction[0] == 1:
            st.success("This is REAL news ✅")
        else:
            st.error("This is FAKE news ❌")
    else:
        st.warning("Please enter some text")