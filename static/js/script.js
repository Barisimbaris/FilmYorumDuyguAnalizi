document.addEventListener('DOMContentLoaded', () => {
    // Posterleri yükle
    fetch("/static/posters.json")
        .then(res => res.json())
        .then(posters => {
            const rowElements = [
                document.getElementById("row1"),
                document.getElementById("row2"),
                document.getElementById("row3"),
            ];

            rowElements.forEach((row, index) => {
                const track = document.createDocumentFragment();

                // Orijinal sıralı liste
                posters.forEach(filename => {
                    const img = document.createElement("img");
                    img.src = `/static/posters/${filename}`;
                    img.alt = "Film Afişi";
                    img.className = "poster";
                    track.appendChild(img);
                });

                // Sonsuz akış için tekrar aynı sırayı ekle
                posters.forEach(filename => {
                    const img = document.createElement("img");
                    img.src = `/static/posters/${filename}`;
                    img.alt = "Film Afişi";
                    img.className = "poster";
                    track.appendChild(img);
                });

                row.appendChild(track);
            });
        })
        .catch(error => console.error('Poster yükleme hatası:', error));

    // Tahmin butonu işlevselliği
    const predictBtn = document.getElementById('predict-btn');
    if (predictBtn) {
        predictBtn.addEventListener('click', async () => {
            const comment = document.getElementById('comment').value.trim();
            const resultDiv = document.getElementById('result');
            const probDiv = document.getElementById('probabilities');

            if (!comment) {
                alert('Lütfen bir yorum girin!');
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ comment }),
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                    resultDiv.textContent = 'Sonuç: Hata oluştu.';
                    probDiv.textContent = 'Olasılıklar: -';
                    return;
                }

                resultDiv.textContent = `Sonuç: ${data.prediction}`;
                resultDiv.className = 'result';
                if (data.prediction === 'Pozitif') {
                    resultDiv.classList.add('positive');
                } else if (data.prediction === 'Negatif') {
                    resultDiv.classList.add('negative');
                } else {
                    resultDiv.classList.add('neutral');
                }

                const probText = Object.entries(data.probabilities)
                    .map(([key, value]) => `${key}: ${value.toFixed(4)}`)
                    .join(', ');
                probDiv.textContent = `Olasılıklar: ${probText}`;
            } catch (error) {
                console.error('Hata:', error);
                alert('Tahmin sırasında bir hata oluştu.');
                resultDiv.textContent = 'Sonuç: Hata oluştu.';
                probDiv.textContent = 'Olasılıklar: -';
            }
        });
    }
});