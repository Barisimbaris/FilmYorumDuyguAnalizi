import joblib
import re
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Türkçe metin temizleme fonksiyonu
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tahmin fonksiyonu
def predict_sentiment(comment, vectorizer, model):
    try:
        print(f"Giriş yorumu: {comment}")
        cleaned_comment = clean_text(comment)
        print(f"Temizlenmiş yorum: {cleaned_comment}")
        if not cleaned_comment:
            return "Hata: Geçerli bir yorum girin.", {}
        comment_tfidf = vectorizer.transform([cleaned_comment])
        print("TF-IDF vektörleştirme tamamlandı.")
        prediction = model.predict(comment_tfidf)[0]
        probabilities = model.predict_proba(comment_tfidf)[0]
        prob_dict = {model.classes_[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
        print(f"Tahmin: {prediction}, Olasılıklar: {prob_dict}")
        return prediction, prob_dict
    except Exception as e:
        print(f"Tahmin sırasında hata: {e}")
        return f"Hata: Tahmin sırasında sorun oluştu: {e}", {}

# Model ve vektörleştiriciyi yükleme
try:
    if not os.path.exists('tfidf_vectorizer.pkl') or not os.path.exists('sentiment_model.pkl'):
        raise FileNotFoundError("tfidf_vectorizer.pkl veya sentiment_model.pkl dosyası bulunamadı.")
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('sentiment_model.pkl')
    print("Model ve vektörleştirici başarıyla yüklendi.")
except Exception as e:
    print(f"Model veya vektörleştirici yüklenirken hata: {e}")
    raise

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin API'si
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        comment = data.get('comment', '')
        print(f"API isteği alındı: {comment}")
        if not comment:
            return jsonify({'error': 'Lütfen bir yorum girin.'}), 400
        prediction, prob_dict = predict_sentiment(comment, vectorizer, model)
        if "Hata" in prediction:
            return jsonify({'error': prediction}), 500
        return jsonify({'prediction': prediction, 'probabilities': prob_dict})
    except Exception as e:
        print(f"API hatası: {e}")
        return jsonify({'error': f"Tahmin sırasında hata: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)