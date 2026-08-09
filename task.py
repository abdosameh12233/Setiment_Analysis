import joblib
import streamlit as st
import re
clf=joblib.load('IMDP.pkl')
cntvector=joblib.load('Tfidf.pkl')
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
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
negation_words = {'not', 'no', 'nor', "n't", 'never', 'none', 'neither', "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't", "wouldn't", "couldn't", "shouldn't", "can't", "cannot"}
def tag_negation_pos(tokens, scope=4):
    tagged = nltk.pos_tag(tokens)
    result = []
    negate_counter = 0
    for word, pos in tagged:
        if word in negation_words:
            negate_counter = scope
            result.append(word)
            continue
        if negate_counter > 0 and pos.startswith(('JJ', 'VB', 'RB')):
            result.append('not_' + word)
            negate_counter -= 1
        elif negate_counter > 0:
            negate_counter -= 1 
            result.append(word)
        else:
            result.append(word)
    return result
def in_predict(text):
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stopWoRds=stopwords.words('english')
    from nltk.stem import PorterStemmer
    pstem=PorterStemmer()
    r=re.sub(r'<br\s+/>',' ',text)
    r=re.sub(r"[^a-zA-Z0-9']",' ',r)
    r=r.lower()
    r=r.split()
    r=tag_negation_pos(r)
    r=[pstem.stem(word) for word in r if word not in stopWoRds]
    r=' '.join(r)
    x=cntvector.transform([r])
    return x
# Button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence first!")
    else:
        if clf.predict(in_predict(user_input)) == 1:
            st.markdown('<div class="resultp">your feedback is positive</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="resultn">your feedback is negative</div>',unsafe_allow_html=True)
