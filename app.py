import streamlit as st
import joblib
import base64

# Load model and vectorizer
model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("tfidf.pkl")

# -------------- CUSTOM CSS -----------------------
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
        .main-title {
            font-size: 36px !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #1a73e8, #34a853);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-title {
            font-size: 18px;
            color: #555;
            margin-top: -10px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 22px;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# -------------- TITLE -----------------------
st.markdown("<h1 class='main-title'>🔍 Toxic Tweet Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter a message below and the AI will classify it as Toxic / Non-Toxic.</p>", unsafe_allow_html=True)

# -------------- SIDEBAR -----------------------
st.sidebar.header("ℹ️ About This App")
st.sidebar.info("""
This application uses a Machine Learning model trained on a 
Toxic Tweet dataset.  
Input any sentence to check if it contains toxic content.
""")

st.sidebar.write("👨‍💻 **Developer:** You") 

# -------------- INPUT CARD -----------------------
with st.container():
    st.write("### ✏️ Enter Your Message:")
    text = st.text_area(" ", height=120, placeholder="Type something like: 'I hate you' or 'Have a great day!'")

# -------------- PREDICT BUTTON ------------------------
if st.button("🚀 Predict Toxicity", use_container_width=True):
    if text.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.markdown("""
                <div class='result-box' style='color:#d9534f; border-left:8px solid #d9534f;'>
                    🚨 Toxic Message Detected!
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div class='result-box' style='color:#28a745; border-left:8px solid #28a745;'>
                    ✅ Non-Toxic Message
                </div>
            """, unsafe_allow_html=True)
