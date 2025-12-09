# ⚡ EPİAŞ Elektrik Fiyat Tahmini (PTF) Projesi

Bu proje, Makine Öğrenmesi (XGBoost) kullanarak Türkiye Elektrik Piyasası'ndaki Piyasa Takas Fiyatını (PTF) tahmin etmeyi amaçlar.

## 🎯 Proje Amacı
* **Girdi:** Rüzgar, Güneş, Doğalgaz üretim planları (KGÜP), Talep Tahmini ve Dolar Kuru vb.
* **Çıktı:** Yarının saatlik elektrik fiyatı (TL/MWh).
* **Model:** 

## 📂 Veri Seti Yapısı
Veriler EPİAŞ Şeffaflık Platformu'ndan çekilmiş ve temizlenmiştir. 
**Örnek Veri (İlk 5 Satır):**

| Tarih      | Saat  | PTF (TL/MWh) | Yük Tahmin | Doğalgaz | Rüzgar | Dolar_Kuru |
| :---       | :---  | :---         | :---       | :---     | :---   | :---       |
| 2025-01-01 | 00:00 | 2494.00      | 32297.0    | 5753.31  | 2024.78| 35.42      |
| 2025-01-01 | 01:00 | 1799.98      | 30678.0    | 5265.68  | 1885.65| 35.42      |
| 2025-01-01 | 02:00 | 1692.99      | 28892.0    | 5246.68  | 1821.14| 35.42      |
| 2025-01-01 | 03:00 | 2244.99      | 27699.0    | 5154.06  | 1805.30| 35.42      |
| 2025-01-01 | 04:00 | 2400.01      | 27015.0    | 5350.01  | 1741.55| 35.42      |

## 🚀 Kurulum
1. Repoyu klonlayın:
   `git clone https://github.com/kad1r-cbn/epias-elektrik-tahmin.git`
2. Kütüphaneleri yükleyin:
   `pip install -r requirements.txt`
3. Veri setini hazırlayın:
   `python epias_merge_dataset.py`
