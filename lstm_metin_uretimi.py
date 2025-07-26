# BİSMİLLAHİRRAHMANİRRAHİM
print("Bismillah " * 3)

"""
Metin üretimi
Bu kod, LSTM (Long Short-Term Memory) modelini kullanarak metin üretimi yapmayı amaçlamaktadır.
Test verisini gpt ile oluşturulmuş bir metinle doldurur ve modelin eğitilmesi için gerekli adımları içerir.
"""

# Gerekli kütüphaneler
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# json dosyasından veriyi yükleme
import json
with open('LSTM/veriseti.json', 'r', encoding='utf-8') as file:
    veri = json.load(file)


# Verideki noktalama işaretlerini ayırma
def noktalama_ayir(metin):
    return re.sub(r'([.,!?;:])', r' \1 ', metin)  # Noktalama işaretlerini ayırarak boşluk ekler


veri = veri["metin"]  # Metin verisini al
veri = [noktalama_ayir(metin) for metin in veri]  # Her metindeki noktalama işaretlerini ayırma
print(veri[:7])  # İlk 7 metni yazdırma


# Metin verisini hazırlama: jetonlama, doldurma, etiket kodlama
# Jetonlama
jetonlayici = Tokenizer(filters='')     # Jetonlayıcı oluşturma, filtreleme yok yani tüm karakterler, noktalama işaretleri kullanılacak
jetonlayici.fit_on_texts(veri)
toplam_jeton = len(jetonlayici.word_index) + 1
#print("\n", jetonlayici.word_index)  # Jeton sözlüğünü yazdırma

# n-gram dizileri oluştur ve doldurma uygula
girdi_dizileri = []
for metin in veri:
    jetonlar = jetonlayici.texts_to_sequences([metin])[0]
    for i in range(1, len(jetonlar)):
        girdi_dizileri.append(jetonlar[:i + 1])

# Doldurma işlemi
maksimum_uzunluk = max(len(dizi) for dizi in girdi_dizileri)
girdi_dizileri = pad_sequences(girdi_dizileri, maxlen=maksimum_uzunluk, padding='pre')  # Pre dememizin sebebi, son elemanı etiket olacak. O yüzden girdi dizilerinin başına sıfır ekliyoruz.
print("\nGirdi dizileri:\n", girdi_dizileri[:7])  # İlk 5 girdi dizisini yazdırma

# Girdi ve etiketleri ayırma
girdi = girdi_dizileri[:, :-1]      # Son eleman hariç tüm elemanlar girdi
etiket = girdi_dizileri[:, -1]      # Son eleman etiket
etiket = tf.keras.utils.to_categorical(etiket, num_classes=toplam_jeton)    # Bunu neden yapıyoruz? Çünkü etiketkler 117, 4 gibi hem yüksek hem düşük değerler içeriyor. Ee bu da eğitimde yüksek olanını daha önemli düşük olanını da önemsiz olarak değerlendiriyor. O yüzden one-hot encoding yapıyoruz.
print("\nGirdi:\n", girdi[:7])     # İlk 10 girdi dizisini yazdırma
print("\nEtiket:\n", etiket[:7])   # İlk 10 etiket dizisini yazdırma

# LSTM modelini oluşturma
model = Sequential()
model.add(Embedding(input_dim=toplam_jeton, output_dim=100, input_length=maksimum_uzunluk - 1))     # Gömme katmanı, jetonları vektörlere dönüştürür
#model.add(LSTM(99, return_sequences=True))
#model.add(Dropout(0.1))     
model.add(LSTM(99))
model.add(Dropout(0.1))
model.add(Dense(toplam_jeton, activation='softmax'))        # Çıkış katmanı, jeton sayısı kadar nöron içerir ve softmax aktivasyon fonksiyonu kullanır

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(girdi, etiket, epochs=100, verbose=1)  # Modeli 100 epoch boyunca eğitiyoruz, verbose=1 ile eğitim sürecini gösteriyoruz

# Modeli kaydetme
model.save('LSTM/metin_uretimi.h5')

# Modeli yükleme (eğer model kaydedildiyse)
model = tf.keras.models.load_model('LSTM/metin_uretimi.h5')

# Modelin özetini yazdırma
model.summary()  # Modelin yapısını ve katmanlarını gösterir

# Metin üretiminde çeşitliği artırmak için bir fonksiyon
def tahmin_yap(tahmin, temperature=1.0):
    """
    Tahmin yapma fonksiyonu. Temperature parametresi ile çeşitliliği artırabiliriz.
    Temperature düşükse (0.1 gibi), en yüksek olasılıklı jeton seçilir.
    Temperature yüksekse (1.0 gibi), daha rastgele seçim yapılır.
    Temmperature çok yüksekse (1.5+ gibi), kaotik, rastgelelik yüksek ama anlamsızlaşabilir.
    0.5-1.0 arası genellikle iyi sonuçlar verir.
    """
    tahmin = np.asarray(tahmin).astype('float64')  # Tahmini float64 tipine dönüştür
    if len(tahmin.shape) == 2:  # Eğer tahmin 2 boyutlu ise (örneğin, birden fazla tahmin varsa)
        tahmin = tahmin[0]
    # Temperature uygulaması
    tahmin = np.log(tahmin) / temperature  # Logaritma ve sıcaklık uygulaması
    tahmin = np.exp(tahmin)  # Ters logaritma
    tahmin /= np.sum(tahmin)  # Olasılıkları normalize et
    jeton_indeksi = np.random.choice(len(tahmin), p=tahmin)  # Olasılıklara göre jeton seçimi
    return jeton_indeksi

# Metin üretme fonksiyonu
def metin_uret(baslangic_metin, uzunluk=50):
    jetonlar = jetonlayici.texts_to_sequences([baslangic_metin])[0]  # Başlangıç metnini jetonlara dönüştür
    jetonlar = pad_sequences([jetonlar], maxlen=maksimum_uzunluk - 1, padding='pre')  # Doldurma uygula
    metin = baslangic_metin  # Başlangıç metnini sakla

    for _ in range(uzunluk):
        tahmin = model.predict(jetonlar, verbose=0)  # Modelden tahmin al
        #jeton_indeksi = np.argmax(tahmin, axis=-1)[0]  # En yüksek olasılıklı jetonu al
        temperature = 0.5 + (_ / uzunluk) * 0.5  # örnek: 0.5 → 1.0 arası artış
        jeton_indeksi = tahmin_yap(tahmin[0], temperature=temperature)  # Tahmin yap, sıcaklık ile çeşitliliği artır
        jeton = jetonlayici.index_word.get(jeton_indeksi, '')  # Jetonu kelimeye dönüştür, bilinmeyen jetonları boş bırak
        if jeton in ['.', '!', '?', ',', ';', ':']:
            metin += jeton      # Eğer jeton noktalama işareti ise kelimeye bitişik yap
            if jeton in ['.', '!', '?']:
                break       # Bazı noktalama işaretleri ile metni sonlandır
        metin += ' ' + jeton  # Metne ekle
        jetonlar = np.append(jetonlar[:, 1:], [[jeton_indeksi]], axis=1)  # Yeni jetonu girdi dizisine ekle

    return metin

# Metin üretme örneği
baslangic_metin = "Haberlere baktığımda"     # "Bu cuma", "Yarın", "Bu hafta", "Hayırlı", "Yarın", "Eve giderken", "Haberlere baktığımda"
metin = metin_uret(baslangic_metin, uzunluk=10)
print("\n", metin)
# Sonuç1: "Bu hafta sonu bir ayet dikkatimi çekti çekti çekti çekti değilim doğru değilim"      # np.argmax ile yapıldı
# Sonuç2: "Bu cuma sabahı iyilik uğrayacağım i̇yi halkasına çamaşırları okurum düzenli sabahlar sevdiğim"
# Sonuç3: "Bu cuma hutbesi çok etkileyiciydi.", " Yarın sabah çay içmeyi seviyorum." "Bu hafta piknik yapmayı planlıyoruz.", "Hayırlı sabahlar , Allah* işlerinizde kolaylık versin."
# *: Allah lafzını küçük yazdı.
# Sonuç4: "Yarın hava durumuna baktın mı?", "Eve giderken iftar bir fincan türk kahvesi iyi gider.", Haberlere baktığımda bitirmek için bütün gün uyukluyor.
