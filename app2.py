import joblib
import re
import os
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)

# Türkçe metin temizleme fonksiyonu (TF-IDF için)
def clean_text_tfidf(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Türkçe metin temizleme fonksiyonu (BERT için, minimal)
def clean_text_bert(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# TF-IDF tahmin fonksiyonu
def predict_sentiment_tfidf(comment, vectorizer, model):
    try:
        print(f"TF-IDF - Giriş yorumu: {comment}")
        cleaned_comment = clean_text_tfidf(comment)
        print(f"TF-IDF - Temizlenmiş yorum: {cleaned_comment}")
        if not cleaned_comment:
            return "Hata: Geçerli bir yorum girin.", {}
        comment_tfidf = vectorizer.transform([cleaned_comment])
        print("TF-IDF - Vektörleştirme tamamlandı.")
        prediction = model.predict(comment_tfidf)[0]
        probabilities = model.predict_proba(comment_tfidf)[0]
        prob_dict = {model.classes_[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)}
        print(f"TF-IDF - Tahmin: {prediction}, Olasılıklar: {prob_dict}")
        return prediction, prob_dict
    except Exception as e:
        print(f"TF-IDF - Tahmin sırasında hata: {e}")
        return f"Hata: Tahmin sırasında sorun oluştu: {e}", {}

# BERT tahmin fonksiyonu
def predict_sentiment_bert(comment, tokenizer, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        print(f"BERT - Giriş yorumu: {comment}")
        cleaned_comment = clean_text_bert(comment)
        print(f"BERT - Temizlenmiş yorum: {cleaned_comment}")
        if not cleaned_comment:
            return "Hata: Geçerli bir yorum girin.", {}
        model.to(device)
        model.eval()
        encoding = tokenizer(
            cleaned_comment,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction_idx = np.argmax(probabilities)
        label_map = {0: 'Negatif', 1: 'Nötr', 2: 'Pozitif'}
        prob_dict = {label_map[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)}
        prediction = label_map[prediction_idx]
        print(f"BERT - Tahmin: {prediction}, Olasılıklar: {prob_dict}")
        return prediction, prob_dict
    except Exception as e:
        print(f"BERT - Tahmin sırasında hata: {e}")
        return f"Hata: Tahmin sırasında sorun oluştu: {e}", {}

# TF-IDF model ve vektörleştiriciyi yükleme
try:
    tfidf_model_path = 'models/sentiment_model.pkl'
    tfidf_vectorizer_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(tfidf_model_path) or not os.path.exists(tfidf_vectorizer_path):
        raise FileNotFoundError("TF-IDF model veya vektörleştirici dosyası bulunamadı.")
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    tfidf_model = joblib.load(tfidf_model_path)
    print("TF-IDF model ve vektörleştirici başarıyla yüklendi.")
except Exception as e:
    print(f"TF-IDF model yüklenirken hata: {e}")
    raise

# BERT model ve tokenizer'ı yükleme
try:
    bert_model_path = 'models/bert_model_3epoch'
    if not os.path.exists(bert_model_path):
        raise FileNotFoundError(f"BERT model klasörü bulunamadı: {bert_model_path}")
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
    print("BERT model ve tokenizer başarıyla yüklendi.")
except Exception as e:
    print(f"BERT model yüklenirken hata: {e}")
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

        # TF-IDF tahmini
        tfidf_prediction, tfidf_prob_dict = predict_sentiment_tfidf(comment, tfidf_vectorizer, tfidf_model)
        if "Hata" in tfidf_prediction:
            return jsonify({'error': tfidf_prediction}), 500

        # BERT tahmini
        bert_prediction, bert_prob_dict = predict_sentiment_bert(comment, bert_tokenizer, bert_model)
        if "Hata" in bert_prediction:
            return jsonify({'error': bert_prediction}), 500

        # Sonuçları birleştir
        response = {
            'tfidf': {
                'prediction': tfidf_prediction,
                'probabilities': tfidf_prob_dict
            },
            'bert': {
                'prediction': bert_prediction,
                'probabilities': bert_prob_dict
            }
        }
        return jsonify(response)
    except Exception as e:
        print(f"API hatası: {e}")
        return jsonify({'error': f"Tahmin sırasında hata: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)