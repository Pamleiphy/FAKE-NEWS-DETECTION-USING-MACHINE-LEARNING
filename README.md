# FAKE-NEWS-DETECTION-USING-MACHINE-LEARNING

Detect whether a news article is real or fake using Machine Learning and NLP techniques.
This project uses Logistic Regression and TF-IDF for text classification and is deployed with Streamlit for a simple web interface.


Main steps:
Data collection
Text preprocessing (NLP)
TF-IDF feature extraction
Model training (Logistic Regression)
Web deployment via Streamlit


✅ Features
Classify news as REAL or FAKE
Simple web app using Streamlit
Runs locally in Jupyter or as a standalone app
High accuracy on test data (≈98%)
Lightweight and easy to extend


📊 Dataset
Source: Kaggle Fake and Real News Dataset
Fake and Real News Dataset on Kaggle

Data Files:
Fake.csv — fake news articles
True.csv — real news articles

Each article contains:
Title
Text
Subject
Date


⚙️ Tech Stack
Programming Language: Python

Libraries:
Pandas
NumPy
scikit-learn
NLTK
Streamlit

IDE:
Jupyter Notebook
VS Code


💻 Installation
Clone this repository:
bash
Copy
Edit
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Ensure the dataset files (Fake.csv and True.csv) are in your project directory.


🚀 Usage
1. Run in Jupyter Notebook
Open the notebook:
Copy
Edit
Fake_News_Detection.ipynb
Execute cells step-by-step to train the model and test predictions.


3. Run the Web App
Start Streamlit:
bash
Copy
Edit
streamlit run app.py
Then open your browser at:
arduino
Copy
Edit
http://localhost:8501
Paste a news article into the text box and click Predict!


📈 Results
Model Performance:

Accuracy: 98%

Precision: 99%

Recall: 98%


🛠️ Future Improvements

Deploy online (Streamlit Cloud, Hugging Face Spaces, etc.)

Support for multiple languages

Integration with live news feeds or APIs

Highlight words contributing to prediction

Experiment with deep learning (LSTM, BERT)
