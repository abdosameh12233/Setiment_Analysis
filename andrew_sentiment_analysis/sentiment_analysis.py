import joblib
import streamlit as st
import re
import pickle
import numpy as np

with open('data.pkl', 'rb') as file:
    freqs = pickle.load(file)
LR=joblib.load('model.pkl')
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 30px;
        color: darkblue;
        margin-bottom: 20px;
    }
    .resultp {
        font-size: 24px;
        font-weight: bold;
        color: darkgreen;
    }
    .resultn {
        font-size: 24px;
        font-weight: bold;
        color: darkred;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Sentiment Analysis Website</div>', unsafe_allow_html=True)

# Text input
user_input = st.text_area("Enter a sentence:")
# cleaning text input and tokenizing it





import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

# Download the necessary NLTK resource packages if you don't have them
nltk.download('stopwords')


stemmer = PorterStemmer()
stopwords_english = stopwords.words('english')
    
    # 1. Remove old style retweet text "RT"
tweet = re.sub(r'^RT[\s]+', '', user_input)

    # 2. Remove hyperlinks
tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)

    # 3. Remove hashtags (only removes the '#' sign, keeps the word text)
tweet = re.sub(r'#', '', tweet)

    # 4. Remove stock market tickers like $GE
tweet = re.sub(r'\$\w*', '', tweet)
tweet = re.sub(r'<br />', '', tweet)

    # 5. Tokenize the tweet text
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tweet_tokens = tokenizer.tokenize(tweet)

tweets_clean = []
for word in tweet_tokens:
    if (word not in stopwords_english and  # Remove stopwords
        word not in string.punctuation):   # Remove punctuation marks
            
            # Stem the word and append to clean list
        stemmed_word = stemmer.stem(word)
        tweets_clean.append(stemmed_word)
# convert tokenized text to 2 values [pos,neg]
pos = 0
neg = 0
for i in tweets_clean:
    pos = pos + freqs.get((i,1),0)
    neg = neg + freqs.get((i,0),0)

# Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence first!")
    else:
        if LR.predict(np.array([[pos,neg]])) == 1:
            st.markdown('<div class="resultp">your feedback is positive</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="resultn">your feedback is negative</div>',unsafe_allow_html=True)
