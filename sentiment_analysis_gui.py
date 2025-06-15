import joblib
import re
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sklearn
import warnings

# scikit-learn sürüm kontrolü
required_sklearn_version = "1.6.1"
if sklearn.__version__ != required_sklearn_version:
    warnings.warn(
        f"Uyarı: scikit-learn {sklearn.__version__} kullanılıyor, ancak model {required_sklearn_version} ile oluşturuldu. "
        "Lütfen `pip install scikit-learn==1.6.1` komutunu çalıştırın."
    )

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

# Tkinter arayüzü
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Türkçe Film Yorumu Duygu Analizi")
        self.root.geometry("650x500")
        self.root.resizable(False, False)

        # Gradyan arka plan
        self.canvas = tk.Canvas(root, width=650, height=500, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.create_gradient()

        # Stil ayarları
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10, relief="raised")
        style.configure("TLabel", background="transparent", font=("Helvetica", 11))

        # Çerçeve (3D efekt)
        self.frame = tk.Frame(self.canvas, bg="#ffffff", bd=5, relief="raised")
        self.frame.place(relx=0.5, rely=0.5, anchor="center", width=550, height=400)

        # Başlık
        self.title_label = ttk.Label(
            self.frame, text="Duygu Analizi", font=("Helvetica", 20, "bold"), foreground="#2c3e50"
        )
        self.title_label.pack(pady=20)

        # Yorum giriş alanı
        self.comment_label = ttk.Label(self.frame, text="Yorumunuzu Girin:", font=("Helvetica", 12, "bold"))
        self.comment_label.pack(pady=5)

        self.comment_text = tk.Text(self.frame, height=4, width=50, font=("Helvetica", 11), relief="sunken", bd=3)
        self.comment_text.pack(pady=10)
        self.comment_text.insert(tk.END, "Örnek: Film harikaydı, bayıldım!")

        # Tahmin butonu
        self.predict_button = ttk.Button(self.frame, text="Tahmin Et", command=self.make_prediction)
        self.predict_button.pack(pady=15)

        # Sonuç alanı
        self.result_label = ttk.Label(
            self.frame, text="Sonuç: Henüz tahmin yapılmadı.", font=("Helvetica", 12, "bold"), wraplength=500
        )
        self.result_label.pack(pady=10)

        # Olasılıklar alanı
        self.prob_label = ttk.Label(
            self.frame, text="Olasılıklar: -", font=("Helvetica", 10), wraplength=500
        )
        self.prob_label.pack(pady=5)

    def create_gradient(self):
        # Gradyan arka plan (mavi tonları)
        for i in range(500):
            r = int(200 - (i / 500) * 100)
            g = int(220 - (i / 500) * 80)
            b = int(255 - (i / 500) * 50)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, 650, i, fill=color)

    def make_prediction(self):
        comment = self.comment_text.get("1.0", tk.END).strip()
        print(f"Kullanıcı girişi: {comment}")
        if not comment or comment == "Örnek: Film harikaydı, bayıldım!":
            messagebox.showwarning("Uyarı", "Lütfen geçerli bir yorum girin!")
            return

        prediction, prob_dict = predict_sentiment(comment, vectorizer, model)
        print(f"Arayüz güncelleme: Tahmin: {prediction}, Olasılıklar: {prob_dict}")
        if "Hata" in prediction:
            messagebox.showerror("Hata", prediction)
            return

        # Sonuçları güncelleme
        self.result_label.config(text=f"Sonuç: {prediction}")
        prob_text = ", ".join([f"{k}: {v:.4f}" for k, v in prob_dict.items()])
        self.prob_label.config(text=f"Olasılıklar: {prob_text}")

        # Renk değiştirme
        colors = {"Pozitif": "#2ecc71", "Negatif": "#e74c3c", "Nötr": "#7f8c8d"}
        self.result_label.config(foreground=colors.get(prediction, "#000000"))

# Tkinter uygulamasını başlatma
if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()