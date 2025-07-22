import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.stem import SnowballStemmer


df = pd.read_csv("/home/ono/nlp/sms/spam.csv", encoding="latin1")

#temizlemek icin
def clean_text(text_input):
    text_input = str(text_input).lower()#kucuk harf
    text_input = re.sub(r'[^\w\s]', '', text_input) #noktalama
    text_input = re.sub(r'\d+', '', text_input) #sayilar
    text_input = " ".join(text_input.split()) #bosluklar
    stop_words = text.ENGLISH_STOP_WORDS
    text_input = " ".join([word for word in text_input.split() if word not in stop_words])
    stemmer = SnowballStemmer("english")
    text_input = " ".join([stemmer.stem(word) for word in text_input.split()])
    return text_input
    
df["cleaned"] = df.v2.apply(clean_text)
    
    
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df.cleaned)

feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1

df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score":tfidf_score})

df_tfidf_sorted = df_tfidf.sort_values(by="tfidf_score", ascending=False)
print(df_tfidf_sorted.head(10))

