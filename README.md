# 🧠 Natural Language Processing (NLP) Projects

Bu repoda, doğal dil işleme (NLP) konularına odaklanarak gerçekleştirdiğim Python projelerini bulabilirsiniz. Projeler, temel metin madenciliği tekniklerini içermekte ve çeşitli veri setleriyle uygulanmıştır.

---

## 📁 Projeler

### 1. 🎬 IMDB Film Yorumları – Bag of Words (BoW)

- **Açıklama:**  
  IMDB film yorumları veri seti kullanılarak yorumların pozitif mi negatif mi olduğunu sınıflandırmak için `Bag of Words` (BoW) tekniği uygulanmıştır.

- **Kullanılan Teknikler:**
  - Veri temizleme (noktalama işaretleri, stopwords temizliği)
  - BoW vektörleştirme (`CountVectorizer`)
  - Naive Bayes sınıflandırıcı

- **Amaç:**  
  Basit kelime sıklığına dayalı bir model ile metin sınıflandırmayı denemek.

- **DATASET:**  
 https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

### 2. 📱 SMS Spam Tespiti – TF-IDF

- **Açıklama:**  
  SMS mesajları üzerinden spam olup olmadığını tahmin etmek için `TF-IDF` (Term Frequency - Inverse Document Frequency) yöntemiyle özellik çıkarımı yapılmıştır.

- **Kullanılan Teknikler:**
  - Metin ön işleme (küçük harfe çevirme, gereksiz karakter temizliği)
  - TF-IDF vektörleştirme (`TfidfVectorizer`)

- **Amaç:**  
  Kelimelerin önemini bağlama göre değerlendiren TF-IDF tekniğiyle spam tespiti yapmak.

- **DATASET:**
 https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

---

## ⚙️ Kullanılan Kütüphaneler

```bash
scikit-learn
pandas
numpy
nltk
