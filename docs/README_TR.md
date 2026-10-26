# MIZAN motor sıcaklığı kestirimi

Bu depo, bir PMSM motorun kaydedilmiş mıknatıs sıcaklığı referansını yedi çevrimiçi kanaldan kestiren çalışır modeli sunar. Gerçek eğitilmiş ağırlıklar, Python paketi, CSV aracı ve klasik karşılaştırmalar dahildir. Çıkarım için NumPy, PyTorch veya scikit-learn gerekmez.

```bash
python -m pip install -e .
python -m mizan_motor predict --model models/p17.json --input examples/telemetry.csv --output predictions.csv
python scripts/verify_release.py
```

Girdiler: ortam ve soğutucu sıcaklıkları, d/q gerilimleri, d/q akımları ve motor hızı. Örnekler kaydedilmiş 2 Hz hızında ve zaman sırasında gelmelidir. Her ayrı sürüşte geçmiş sıfırlanır. Hedef rotor sıcaklığı, ilk rotor sıcaklığı, stator sıcaklıkları ve gelecek örnekler kullanılmaz.

Üç ayrılmış test sürüşünde eşit sürüş ağırlıklı MAE **1,641 °C** oldu. Ayrı dondurulmuş SVR + Ridge **1,609 °C** ile daha düşük hata verdi. Dolayısıyla bu deney bütün güçlü klasiklere üstünlük veya yeni algoritma kanıtı değildir. Ağın en büyük eksik tahmini başlangıçta **17,536 °C**; bu satırlar sonuçtan çıkarılmadı.

Ölçümler tek motor ve tezgâhtandır. Hedef kaydedilmiş sensör referansıdır; bütün rotorun gerçek maksimumu veya güvenli sürüş kararı değildir. Yeni motor aktarımı ve üretici karar eşiği bu çalışmada doğrulanmadı.

`examples/telemetry.csv`, önceden değerlendirilmiş her test sürüşünün ilk 12 satırından oluşur. Tahmin uyumu kontrolü içindir; yeni test veya ortalama doğruluk örneklemesi değildir. Tam veri depoya dahil edilmedi. Yayına hazırlık sırasında yeniden eğitim yapılmadı.

Kod Apache-2.0; ağırlıklar, küçük ölçüm örnekleri, grafik ve veri kaynaklı sonuçlar CC BY-SA 4.0 kapsamındadır. Kaynak veri Kirchgässner, Wallscheid ve Böcker'in [Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) veri kümesidir. Ayrıntılar: [model kartı](model-card.md), [veri kaydı](data.md), [kaynak ve sağlama değerleri](provenance.json).
