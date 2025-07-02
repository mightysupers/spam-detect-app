import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Sample training data
messages = [
    "Congratulations! You've won a free ticket",
    "You have been selected for a $1000 gift",
    "Hi, how are you doing today?",
    "Reminder: your meeting is at 10 AM",
    "Get rich fast by clicking this link",
    "Free access to your reward now!",
    "Are we still on for lunch tomorrow?",
    "Don't forget to submit your assignment"
]

labels = [1, 1, 0, 0, 1, 1, 0, 0]  # 1 = Spam, 0 = Ham

# 2. TF-IDF Vectorization
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(messages)

# 3. Model training
model = MultinomialNB()
model.fit(X, labels)

# 4. Streamlit interface
st.title("📩 Simple Spam Detector")

user_input = st.text_input("Enter a message to check:")

if user_input:
    vector = tfidf.transform([user_input])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        st.error("🚨 This message is likely SPAM.")
    else:
        st.success("✅ This message is likely HAM (Not spam).")
