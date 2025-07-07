import streamlit as st
import pickle
import re, string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean function
stop = set(stopwords.words('english'))
def clean(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join([w for w in text.split() if w not in stop])

# ----------------- Streamlit Theme -----------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #111;
    }
    .main {
        background-color: #111111;
        color: #f5f5f5;
    }
    .stTextArea textarea {
        background-color: #222;
        color: #f5f5f5;
        font-size: 16px;
    }
    .stButton button {
        background-color: #ff6f00;
        color: white;
        font-weight: bold;
    }
    .stTitle, .stMarkdown h1, h2, h3 {
        color: #ff6f00 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.markdown("## 🧠 About")
    st.write("This tool uses **machine learning** to detect whether a news article is **real** or **fake** based on its content.")

# Title
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detector</h1>", unsafe_allow_html=True)

# Text input
text = st.text_area("📝 Paste a news article here:", height=200)

# Predict when button is clicked
if st.button("🔍 Predict"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        cleaned = clean(text)
        vect = vectorizer.transform([cleaned])

        if vect.nnz == 0:
            st.error("⚠️ The model could not understand this article. Try entering a longer or more detailed article.")
        else:
            pred = model.predict(vect)
            proba = model.predict_proba(vect)[0]
            label = "REAL" if pred[0] == 1 else "FAKE"
            confidence = round(proba[pred[0]] * 100, 2)

            if label == "REAL":
                st.success(f"✅ This is REAL news. (Confidence: {confidence}%)")
            else:
                st.error(f"🚨 This is FAKE news. (Confidence: {confidence}%)")
