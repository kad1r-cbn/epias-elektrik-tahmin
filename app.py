# =============================================================================
# EPİAŞ AI TRADER PREMIUM - MARJİNAL & MİNİMALİST TASARIM
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import datetime
import holidays
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. PREMIUM SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EPİAŞ PTF Tahmin Modeli",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar'ı kapalı başlatıyoruz, daha clean
)

# Özel CSS (Marjinal Tasarım)
st.markdown("""
<style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    /* Metrik Kartları */
    div[data-testid="metric-container"] {
        background-color: #1E212B;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: scale(1.02);
        border-color: #00D4FF;
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Buton */
    .stButton>button {
        background: linear-gradient(90deg, #00C6FF 0%, #0072FF 100%);
        color: white;
        border: none;
        border-radius: 25px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #0072FF;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATA LOAD & ENGINE (GİZLİ KAHRAMAN)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    model_path = os.path.join('models', 'epias_model_final.pkl')
    data_path = os.path.join('data_s', 'data_set_ex.xlsx')

    try:
        model_pkg = joblib.load(model_path)
        df = pd.read_excel(data_path)

        # Hızlı temizlik
        df.columns = [col.strip() for col in df.columns]
        df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')

        # Saat düzeltme
        if 'Saat' in df.columns:
            if df['Saat'].dtype == 'O':
                df['Saat_Int'] = df['Saat'].astype(str).str.split(':').str[0].astype(int)
            else:
                df['Saat_Int'] = df['Saat']
        else:
            df['Saat_Int'] = df['Tarih'].dt.hour

        # Para birimi temizliği
        for col in ['PTF (TL/MWH)', 'Yük Tahmin Planı (MWh)']:
            if col in df.columns and df[col].dtype == 'O':
                df[col] = df[col].apply(
                    lambda x: float(str(x).replace('.', '').replace(',', '.')) if isinstance(x, str) else x)

        return model_pkg, df
    except Exception as e:
        return None, None


model_package, df_history = load_engine()

if not model_package:
    st.error("Sistem Yüklenemedi! Lütfen training.py dosyasını çalıştırın.")
    st.stop()

model = model_package['model']
feature_list = model_package['features']


# Otomatik Veri Çekici
def get_auto_data(tarih, saat):
    # Veri setinde ara
    mask = (df_history['Tarih'] == pd.to_datetime(tarih)) & (df_history['Saat_Int'] == saat)
    row = df_history[mask]

    if not row.empty:
        d = row.iloc[0]
        return d
    else:
        # Yoksa ortalama al
        m = pd.to_datetime(tarih).month
        mask_m = (df_history['Tarih'].dt.month == m) & (df_history['Saat_Int'] == saat)
        return df_history[mask_m].mean(numeric_only=True)


def get_lag_prices(tarih, saat):
    # Basitleştirilmiş Lag bulucu
    try:
        dt = pd.to_datetime(tarih)
        d_prev = dt - datetime.timedelta(days=1)
        w_prev = dt - datetime.timedelta(days=7)

        mask_d = (df_history['Tarih'] == d_prev) & (df_history['Saat_Int'] == saat)
        mask_w = (df_history['Tarih'] == w_prev) & (df_history['Saat_Int'] == saat)

        p_24 = df_history[mask_d]['PTF (TL/MWH)'].values[0] if not df_history[mask_d].empty else 2000
        p_168 = df_history[mask_w]['PTF (TL/MWH)'].values[0] if not df_history[mask_w].empty else 2000
        return p_24, p_168
    except:
        return 2000, 2000


# -----------------------------------------------------------------------------
# 3. MİNİMALİST ARAYÜZ (GİRİŞ KISMI)
# -----------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(
        "foto/epias.jpg",
        width=80)
with col_title:
    st.title("EPİAŞ INTELLIGENCE")
    st.caption("AI Powered Energy Market Forecasting")

st.markdown("---")

# Seçim Alanı (Tek satırda basit seçim)
c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    secilen_tarih = st.date_input("Analiz Tarihi", datetime.date.today())
with c2:
    secilen_saat = st.selectbox("Saat Dilimi", list(range(24)), index=14)
with c3:
    st.write("")  # Boşluk
    st.write("")
    btn_predict = st.button("ANALİZİ BAŞLAT")

# -----------------------------------------------------------------------------
# 4. TAHMİN MOTORU VE GÖRSELLEŞTİRME
# -----------------------------------------------------------------------------
if btn_predict:
    # Verileri Arkada Topla (Kullanıcı Görmez)
    raw_data = get_auto_data(secilen_tarih, secilen_saat)
    lag_24, lag_168 = get_lag_prices(secilen_tarih, secilen_saat)

    # Model Input Hazırla
    input_df = pd.DataFrame([0], columns=['dummy'])

    # Tarihsel Dönüşümler
    dt = pd.to_datetime(f"{secilen_tarih} {secilen_saat}:00:00")
    input_df['Hour_Sin'] = np.sin(2 * np.pi * secilen_saat / 24)
    input_df['Hour_Cos'] = np.cos(2 * np.pi * secilen_saat / 24)
    input_df['Day_Sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
    input_df['Day_Cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
    input_df['Is_Weekend'] = 1 if dt.dayofweek >= 5 else 0
    input_df['Is_Holiday'] = 1 if dt in holidays.TR() else 0

    # Lag & Feature Engineering
    input_df['PTF_Lag_24'] = lag_24
    input_df['PTF_Lag_168'] = lag_168
    rm = (lag_24 + lag_168) / 2
    input_df['PTF_Roll_Mean_24'] = lag_24
    input_df['PTF_Roll_Mean_168'] = rm
    input_df['PTF_Roll_Std_24'] = 50
    input_df['Relative_Price_Pos'] = (lag_24 - rm) / (rm + 1)
    input_df['Price_Momentum'] = lag_24 - lag_168

    # Üretim Verileri (Otomatik Gelenler)
    # Eğer veri yoksa varsayılan ata
    yuk = raw_data.get('Yük Tahmin Planı (MWh)', 35000) if raw_data is not None else 35000
    ruzgar = raw_data.get('Rüzgar', 3000) if raw_data is not None else 3000
    gunes = raw_data.get('Güneş', 0) if raw_data is not None else 0
    dogalgaz = raw_data.get('Doğalgaz', 8000) if raw_data is not None else 8000
    komur = raw_data.get('İthal Kömür', 5000) if raw_data is not None else 5000

    ren = ruzgar + gunes
    input_df['Total_Renewable_Lag24'] = ren
    input_df['Net_Load'] = yuk - ren
    therm = dogalgaz + komur
    input_df['Total_Thermal_Lag24'] = therm
    input_df['Thermal_Stress'] = therm / (yuk + 1)

    # Shifted Columns
    input_df['Doğalgaz_Lag24'] = dogalgaz
    input_df['Rüzgar_Lag24'] = ruzgar
    input_df['Güneş_Lag24'] = gunes
    input_df['İthal Kömür_Lag24'] = komur

    # Eksikleri 0 yap
    for c in feature_list:
        if c not in input_df.columns: input_df[c] = 0

    # TAHMİN
    pred = model.predict(input_df[feature_list])[0]
    pred = max(0, pred)

    # =========================================================================
    # GÖRSEL ŞÖLEN (MODERN TASARIM)
    # =========================================================================

    st.markdown("### 📊 Piyasa Özeti")

    # 1. BÜYÜK KARTLAR (METRICS)
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Tahmini Fiyat (PTF)", f"{pred:,.2f} ₺", delta=f"{pred - lag_24:.1f} ₺ (Düne Göre)")
    m2.metric("Sistem Yükü", f"{yuk:,.0f} MWh", delta="Talep Durumu", delta_color="off")
    m3.metric("Yenilenebilir Enerji", f"{ren:,.0f} MWh", f"%{(ren / yuk) * 100:.1f} Pay")
    m4.metric("Risk Skoru (Volatilite)", "Düşük" if pred < 2000 else "Yüksek", delta_color="inverse")

    st.write("")
    st.write("")

    # 2. HIZ GÖSTERGESİ (GAUGE CHART) - Herkesin Anlayacağı Grafik
    # Fiyatın ucuz mu pahalı mı olduğunu gösterir

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fiyat Basınç Göstergesi", 'font': {'size': 24, 'color': "white"}},
        number={'suffix': " TL", 'font': {'color': "white"}},
        gauge={
            'axis': {'range': [0, 4000], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00D4FF"},
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1500], 'color': '#00ff00'},  # Ucuz (Yeşil)
                {'range': [1500, 2500], 'color': '#ffff00'},  # Normal (Sarı)
                {'range': [2500, 4000], 'color': '#ff0000'}],  # Pahalı (Kırmızı)
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': pred}}))

    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})

    # 3. PASTA GRAFİK (ENERJİ KAYNAKLARI) - Basit ve Renkli
    labels = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Kömür', 'Diğer']
    values = [ruzgar, gunes, dogalgaz, komur, max(0, yuk - (ruzgar + gunes + dogalgaz + komur))]

    fig_pie = px.pie(names=labels, values=values, title='Enerji Üretim Dağılımı', hole=0.5,
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, showlegend=True)

    # Grafikleri Yan Yana Koy
    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    with g2:
        st.plotly_chart(fig_pie, use_container_width=True)

    # 4. YORUM KARTI (ALGORİTMA KONUŞUYOR)
    st.info(f"💡 **AI Analizi:** Sistem yükünün %{(ren / yuk) * 100:.1f}'si yenilenebilir kaynaklardan karşılanıyor. "
            f"Fiyatlar {'beklenen seviyenin altında' if pred < 2200 else 'yüksek seyrediyor'}. "
            f"Termik santraller {'baskı altında' if therm / yuk > 0.5 else 'rahat çalışıyor'}.")

else:
    # Başlangıçta boş durmasın diye havalı bir placeholder
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 50px;'>
        <h2>Hazır olduğunda "ANALİZİ BAŞLAT" butonuna bas.</h2>
        <p>Yapay zeka motoru EPİAŞ verilerini taramak için bekliyor...</p>
    </div>
    """, unsafe_allow_html=True)