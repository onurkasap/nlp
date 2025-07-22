#import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

df = pd.read_csv("/home/ono/nlp/imdb/IMDB Dataset.csv")

# metin verileri
documents = df["review"]
labels = df["sentiment"]

#metin temizleme
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+","" , text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join([word for word in text.split() if len(word)>5])
    return text

cleaned_doc = [clean_text(row) for row in documents]

#bow islemleri
vectorizer = CountVectorizer()

#metni sayisal hale getirme
X = vectorizer.fit_transform(cleaned_doc)

#kelime kumesi
feature_names = vectorizer.get_feature_names_out()


#kelime frekanslari
word_counts = X.sum(axis=0).A1
word_freq = dict(zip(feature_names, word_counts))

#en sik gecen 20 kelime
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
print("En sik gecen 20 kelime: ")
for word, freq in sorted_words[:20]:
    print(f"{word}:  {freq}")


