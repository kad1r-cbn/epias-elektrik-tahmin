# =============================================================================
# EPİAŞ ELEKTRİK PİYASASI TAHMİN ASİSTANI (STREAMLIT APP)
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import datetime

# -----------------------------------------------------------------------------
# 1. SAYFA AYARLARI (GÖRSEL MAKYAJ)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EPİAŞ AI Trader",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS (Daha profesyonel görünüm için)
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box_shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MODELİ YÜKLEME (CACHE MEKANİZMASI)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'epias_model_final.pkl')
    try:
        package = joblib.load(model_path)
        return package
    except FileNotFoundError:
        st.error("🚨 Model dosyası bulunamadı! Lütfen önce 'training.py' dosyasını çalıştırın.")
        return None


# Modeli yükle
model_package = load_model()

if model_package:
    model = model_package['model']
    feature_list = model_package['features']

    # Modelin içindeki best_params'ı al (eğer varsa)
    best_params = model_package.get('best_params', {})
else:
    st.stop()  # Model yoksa uygulamayı durdur

# -----------------------------------------------------------------------------
# 3. YAN MENÜ (SIDEBAR) - GİRDİLER
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://www.epias.com.tr/wp-content/uploads/2019/06/epias-logo.png", width=200)
    st.title("⚡ Parametre Paneli")
    st.markdown("---")

    # Tarih ve Saat Seçimi
    secilen_tarih = st.date_input("Tahmin Tarihi", datetime.date.today() + datetime.timedelta(days=1))
    secilen_saat = st.slider("Saat Seçimi (0-23)", 0, 23, 14)

    st.markdown("### 🏭 Piyasa Koşulları")

    # Kullanıcıdan Girdiler (Varsayılan değerler ortalama değerlerdir)
    yuk_tahmini = st.number_input("Yük Tahmini (MWh)", min_value=10000, max_value=60000, value=35000, step=500)

    st.markdown("### 🔋 Üretim Senaryosu (MW)")
    ruzgar = st.number_input("Rüzgar Üretimi", 0, 15000, 3000)
    gunes = st.number_input("Güneş Üretimi", 0, 15000, 0 if secilen_saat > 18 or secilen_saat < 6 else 2000)
    dogalgaz = st.number_input("Doğalgaz Üretimi", 0, 15000, 8000)

    # Diğerleri (Ortalama varsayılanlar)
    ithal_komur = st.sidebar.number_input("İthal Kömür (Opsiyonel)", 0, 10000, 5000)
    linyit = st.sidebar.number_input("Linyit (Opsiyonel)", 0, 10000, 4000)

    # Geçmiş Fiyat Bilgisi (Lag için)
    st.markdown("### 💰 Geçmiş Fiyatlar")
    ptf_dun = st.number_input("Dünkü Aynı Saat Fiyatı (PTF)", 0, 5000, 2000)
    ptf_hafta = st.number_input("Geçen Hafta Aynı Saat Fiyatı", 0, 5000, 1900)

    # Tahmin Butonu
    predict_btn = st.button("FİYAT TAHMİN ET 🚀")

# -----------------------------------------------------------------------------
# 4. ANA EKRAN (DASHBOARD)
# -----------------------------------------------------------------------------
st.title("💡 EPİAŞ Elektrik Fiyat Tahmin Modeli")
st.markdown(f"**Seçilen Tarih:** {secilen_tarih.strftime('%d %B %Y')} | **Saat:** {secilen_saat}:00")

# Sekmeler
tab1, tab2 = st.tabs(["📊 Tahmin & Simülasyon", "🧠 Model Analitiği"])

with tab1:
    if predict_btn:
        # --- FEATURE ENGINEERING (CANLI) ---
        # Kullanıcının girdiği verileri modelin anlayacağı dile çeviriyoruz.

        # Tarihsel Özellikler
        tarih_dt = pd.to_datetime(f"{secilen_tarih} {secilen_saat}:00:00")

        # DataFrame Oluştur (Tek satırlık)
        input_data = pd.DataFrame([0], columns=['dummy'])  # Geçici

        # Sniper Değişkenleri Hesapla
        # 1. Döngüsel Zaman
        input_data['Hour_Sin'] = np.sin(2 * np.pi * secilen_saat / 24)
        input_data['Hour_Cos'] = np.cos(2 * np.pi * secilen_saat / 24)
        input_data['Day_Sin'] = np.sin(2 * np.pi * tarih_dt.dayofweek / 7)
        input_data['Day_Cos'] = np.cos(2 * np.pi * tarih_dt.dayofweek / 7)
        input_data['Is_Weekend'] = 1 if tarih_dt.dayofweek in [5, 6] else 0

        # 2. Lag Değişkenleri (Kullanıcıdan aldık)
        input_data['PTF_Lag_24'] = ptf_dun
        input_data['PTF_Lag_168'] = ptf_hafta

        # 3. Sniper Özellikler
        # Relative Price (Ortalama yerine basitçe dünkü fiyatı baz alıyoruz canlıda)
        roll_mean_proxy = (ptf_dun + ptf_hafta) / 2  # Canlıda 168 saat geriye gidemeyeceğimiz için proxy kullanıyoruz
        input_data['PTF_Roll_Mean_168'] = roll_mean_proxy
        input_data['Relative_Price_Pos'] = (ptf_dun - roll_mean_proxy) / (roll_mean_proxy + 1)

        # Net Load
        total_ren = ruzgar + gunes  # Basit yenilenebilir
        input_data['Total_Renewable_Lag24'] = total_ren
        input_data['Net_Load'] = yuk_tahmini - total_ren

        # Thermal Stress
        total_therm = dogalgaz + ithal_komur + linyit
        input_data['Total_Thermal_Lag24'] = total_therm
        input_data['Thermal_Stress'] = total_therm / (yuk_tahmini + 1)

        # Momentum & Volatility
        input_data['Price_Momentum'] = ptf_dun - ptf_hafta
        input_data['Volatility'] = 50  # Varsayılan (Canlıda hesaplamak zor)

        # Diğer Shift Edilmiş Kolonlar (Model 24 saat öncesini istiyor)
        input_data['Doğalgaz_Lag24'] = dogalgaz
        input_data['Rüzgar_Lag24'] = ruzgar
        input_data['Güneş_Lag24'] = gunes
        # ... diğerlerini 0 veya varsayılan geçebiliriz (Eksik özellik hatası almamak için)

        # Modelin beklediği tüm sütunları tamamla
        for col in feature_list:
            if col not in input_data.columns:
                input_data[col] = 0  # Bilinmeyenleri 0 kabul et (Güvenli Liman)

        # Sıralamayı Garantiye Al
        input_data = input_data[feature_list]

        # TAHMİN
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        # --- SONUÇ GÖSTERİMİ ---
        st.success("✅ Tahmin Başarıyla Oluşturuldu!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Tahmini PTF Fiyatı", value=f"{prediction:.2f} TL",
                      delta=f"{prediction - ptf_dun:.2f} TL (Düne Göre)")

        with col2:
            st.metric(label="Net Yük (Talebin Gücü)", value=f"{input_data['Net_Load'].iloc[0]:,.0f} MWh")

        with col3:
            stress = input_data['Thermal_Stress'].iloc[0]
            st.metric(label="Termik Stres Oranı", value=f"%{stress * 100:.1f}", delta_color="inverse")

        # Görsel Yorum
        st.markdown("### 🤖 Yapay Zeka Yorumu:")
        if prediction > 2500:
            st.warning("⚠️ **Yüksek Fiyat Uyarısı:** Sistemde stres yüksek. Gaz santralleri devrede olabilir.")
        elif prediction < 1500:
            st.info("📉 **Düşük Fiyat Beklentisi:** Yenilenebilir enerji (Rüzgar/Güneş) piyasayı rahatlatıyor.")
        else:
            st.write("✅ **Normal Piyasa Koşulları:** Fiyatlar beklenen dengede seyrediyor.")

    else:
        st.info("👈 Tahmin sonucunu görmek için yandaki 'FİYAT TAHMİN ET' butonuna basınız.")
        st.image(
            "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
            caption="Enerji Piyasaları", use_column_width=True)

with tab2:
    st.header("Model Performansı ve İstatistikler")

    # Modelden gelen istatistikleri göster (Eğer kaydettiysek)
    if 'best_params' in model_package:
        st.json(model_package['best_params'])
    else:
        st.write("Model parametreleri bulunamadı.")

    st.markdown("""
    **Model Mimarisi:** XGBoost Regressor  
    **Özellik Mühendisliği:** Sniper Features (Net Load, Thermal Stress, Relative Price)  
    **Validasyon:** Time Series Split (Ekim Train / Kasım Test)
    """)