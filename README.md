# 🧠 LSTM ile Türkçe Metin Üretimi

Bu proje, LSTM (Long Short-Term Memory) sinir ağı kullanılarak Türkçe cümleler üretmeyi amaçlamaktadır. Eğitim verisi olarak GPT destekli yapay metinler kullanılmış, modelin metin üretme yeteneği çeşitli tekniklerle geliştirilmiştir.

## ✨ Özellikler

- 📚 Türkçe dilinde kelime tabanlı metin üretimi
- 🔠 N-gram dizileri ile sıralı model eğitimi
- 🧊 Temperature (sıcaklık) ayarıyla çeşitli metin üretimi
- 💾 Model eğitme, kaydetme ve yükleme
- 🔍 Anlamlı ve dilbilgisel olarak düzgün cümleler üretme

## 🛠️ Kullanılan Teknolojiler

- Python 3.x
- TensorFlow / Keras
- NumPy
- JSON
- Regex (`re`)
- LSTM & Embedding katmanları

## 📂 Proje Yapısı
```
LSTM/
├── metin_uretimi.py # Ana Python dosyası
├── veriseti.json # Eğitim verisi (GPT ile oluşturulmuş metinler)
├── metin_uretimi.h5 # Eğitilmiş model dosyası
└── tokenizer.pkl (isteğe bağlı) # Tokenizer nesnesi (kullanılırsa)
```

## 🧠 Öne Çıkan Başarımlar
Bu proje kapsamında yalnızca metin üretimi yapılmamış, aynı zamanda aşağıdaki önemli noktalar başarıyla gerçekleştirilmiştir:

✅ **Cümle sonunu anlama yetisi:**  
Model, eğitildiği verilerden öğrendiği dil yapısı sayesinde noktalama işaretleri (., ?, !) ile cümle sonunu anlamlı yerlerde bitirmeyi öğrenmiştir.

✅ **Temperature kullanımı ile kontrollü rastgelelik:**  
Üretilen metinlerde temperature (sıcaklık) parametresi kullanılarak hem anlamlılık hem de çeşitlilik dengelenmiştir.  
• *Düşük değerler* → daha tutarlı, tekrar eden yapılar  
• *Yüksek değerler* → yaratıcı ama bazen anlamsız cümleler  
• *Kademeli sıcaklık artışı* → daha doğal cümle uzamaları

✅ **N-gram yapısı ile bağlamsal öğrenme:**  
Eğitim verisi, n-gram'lar şeklinde hazırlanarak kelimeler arası ilişki öğrenilmiştir. Böylece model tek tek kelimeleri değil, bağlamı öğrenmiştir.

✅ **Noktalama duyarlılığı:**  
Jetonlayıcıda filtreleme yapılmadığı için noktalama işaretleri korunmuş, modelin dil bilgisine duyarlı öğrenmesi sağlanmıştır.


## 📈 Model Başarımı

Modelin eğitim sürecinde kaydedilen accuracy ve loss değerleri aşağıdaki gibidir:
```
Epoch 1/100     - accuracy: 0.1194 - loss: 6.5205
Epoch 2/100     - accuracy: 0.1623 - loss: 5.6449
Epoch 3/100     - accuracy: 0.1664 - loss: 5.4198
...
Epoch 50/100    - accuracy: 0.7526 - loss: 1.7481
Epoch 52/100    - accuracy: 0.7868 - loss: 1.5922
...
Epoch 98/100    - accuracy: 0.9239 - loss: 0.3623
Epoch 100/100   - accuracy: 0.9209 - loss: 0.3560
```
🔹 Başlangıçta %12 doğruluk ile başlayan model, 100 epoch sonunda %92 doğruluk oranına ulaşmıştır.  
🔹 Her ne kadar bu, modelin başarılı şekilde öğrendiğini gösterse de veri sayısı az olduğu için aşırı öğrenme(overfitting) sorunu yaşanmaktadır.

## 🎓 Öğrenilenler

- LSTM'nin sıralı verilerdeki gücü
- Tokenization ve n-gram oluşturma
- Embedding katmanları ile kelimeleri sayısal forma dönüştürme
- Temperature ile üretimde rastgelelik kontrolü

## 📌 Notlar

Bu proje temel bir örnektir. Daha iyi sonuçlar için:
- Daha büyük ve çeşitli bir veri kümesi kullanılabilir.
- Bidirectional LSTM veya GRU gibi alternatif modeller denenebilir.
- Türkçe özel karakterler dikkatlice işlenmelidir.
- Attention veya Transformer mimarisi ileride incelenebilir.
