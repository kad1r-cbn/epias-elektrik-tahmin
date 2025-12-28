# ⚡ EPİAŞ Piyasa Takas Fiyatı (PTF) Tahmin Modeli

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-XGBoost%20%7C%20Pandas%20%7C%20Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Proje Hakkında
Bu proje, Türkiye Elektrik Piyasası'ndaki (EPİAŞ) saatlik **Piyasa Takas Fiyatı'nı (PTF)** makine öğrenmesi yöntemleriyle tahmin etmek amacıyla geliştirilmiştir. Enerji piyasalarındaki volatiliteyi öngörmek, üretim planlaması ve ticaret stratejileri için hayati önem taşımaktadır.

Proje kapsamında **Miuul Data Science Bootcamp** bitirme projesi olarak; veri toplama, özellik mühendisliği (feature engineering) ve modelleme süreçleri uçtan uca (end-to-end) uygulanmıştır.

## 🎯 İş Problemi ve Amaç
* **Problem:** Enerji fiyatlarının, arz-talep dengesi ve hammadde maliyetlerine bağlı olarak anlık değişimi.
* **Amaç:** Gelecek 24 saatin elektrik fiyatlarını minimum hata payı ile tahmin ederek stratejik karar mekanizmalarını desteklemek.
* **Hedef Metrik:** Düşük RMSE (Kök Ortalama Kare Hata) ve düşük MAPE (Ortalama Mutlak Yüzde Hata).

## 📂 Veri Seti ve Özellikler
Veriler **EPİAŞ Şeffaflık Platformu**'ndan API aracılığıyla çekilmiş ve temizlenmiştir.

**Kullanılan Temel Değişkenler:**
* **Tarihsel Veriler:** Saatlik PTF, Yük Tahmini (Talep).
* **Üretim Planları (KGÜP):** Rüzgar, Güneş, Doğalgaz, Barajlı Hidroelektrik.
* **Ekonomik Göstergeler:** Dolar Kuru (USD/TRY).
* **Türetilen Özellikler:** Lag Features (Gecikmeli Değişkenler), Rolling Window Statistics (Hareketli Ortalamalar).

| Tarih      | Saat  | PTF (TL/MWh) | Yük Tahmin | Doğalgaz | Rüzgar | Dolar_Kuru |
| :---       | :---  | :---         | :---       | :---     | :---   | :---       |
| 2025-01-01 | 00:00 | 2494.00      | 32297.0    | 5753.31  | 2024.78| 35.42      |
| 2025-01-01 | 01:00 | 1799.98      | 30678.0    | 5265.68  | 1885.65| 35.42      |

## ⚙️ Kullanılan Teknolojiler ve Yöntemler
* **Veri İşleme:** Pandas, NumPy
* **Görselleştirme:** Matplotlib, Seaborn
* **Modelleme:** XGBoost Regressor (Gradient Boosting)
* **Optimizasyon:** GridSearchCV / Optuna (Hiperparametre optimizasyonu için)

## 📊 Model Sonuçları ve Başarı Metrikleri
Test veri seti üzerinde elde edilen model performansı aşağıdadır:

| Metrik | Değer |
| :--- | :--- |
| **RMSE** | **459.02 TL** |
| **MAE** | **%20.26**  |
| **İyileşme Oranı** | **%27.88**  |

> **Analiz:** Model, özellikle volatilite'nin düşük olduğu saatlerde %95+ doğrulukla tahmin yapabilmektedir. ![FORECASTINH](https://github.com/user-attachments/assets/5b7b6e60-1b15-4bbf-9490-d9550bb5aa9a)


### 📈 Tahmin vs Gerçekleşen (Actual vs Predicted)
*(Buraya projenin çıktısı olan bir grafiğin ekran görüntüsünü -screenshot- koymalısın. Görselsiz README olmaz. `` formatında ekle)*

## 🚀 Kurulum ve Kullanım

Projeyi lokalinizde çalıştırmak için:

1.  Repoyu klonlayın:
    ```bash
    git clone [https://github.com/WDG-DS/epias-elektrik-tahmin.git](https://github.com/WDG-DS/epias-elektrik-tahmin.git)
    ```
2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
3.  Veri setini hazırlayın ve modeli çalıştırın:
    ```bash
    python main.py
    ```

## 👥 Takım
* [Kadir](https://github.com/kad1r-cbn)
* [Abdullah Gönül ](https://github.com/apognl)
* [Bilgi Gülçin Sönmez ](https://github.com/bilgigulcinsonmez-dev)
* [Züleyha Erdoğan ](https://github.com/zuleyha-erdogan)

---
*Bu proje Miuul Data Science Bootcamp kapsamında geliştirilmiştir.*
