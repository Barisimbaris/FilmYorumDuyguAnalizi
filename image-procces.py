import os
import json

# Afişlerin bulunduğu klasör
poster_dir = os.path.join('static', 'posters')

# Geçerli görsel uzantıları
valid_ext = {'.jpg', '.jpeg', '.png', '.webp'}

# Görselleri al
posters = [f for f in os.listdir(poster_dir) if os.path.splitext(f)[1].lower() in valid_ext]

# JSON dosyasına yaz
with open(os.path.join('static', 'posters.json'), 'w', encoding='utf-8') as f:
    json.dump(posters, f, ensure_ascii=False, indent=2)

print("posters.json oluşturuldu.")
