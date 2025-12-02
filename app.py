import streamlit as st
import joblib

# Load saved model + vectorizer
model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

st.title("🔍 Toxic Tweet Detector")
st.write("Enter a message and the model will classify it as toxic or non-toxic.")

# Input box
user_input = st.text_area("Enter a text message:")

# Predict button
if st.button("Predict"):
    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)[0]
    
    if prediction == 1:
        st.error("🚨 Toxic Message")
    else:
        st.success("✅ Non-Toxic Message")
