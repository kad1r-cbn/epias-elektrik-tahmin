# =============================================================================
# EPİAŞ INTELLIGENCE PRO - HYBRID TRADING TERMINAL
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
# 1. SAYFA AYARLARI (MARJİNAL TASARIM)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EPİAŞ Pro Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark/Neon Tema CSS
st.markdown("""
<style>
    /* Ana Arka Plan */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    /* Kartlar */
    div[data-testid="metric-container"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    /* Input Alanları */
    .stNumberInput input {
        background-color: #0D1117;
        color: white;
    }
    /* Buton */
    .stButton>button {
        background: linear-gradient(90deg, #238636 0%, #2EA043 100%);
        color: white;
        border: none;
        height: 45px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
        border-radius: 6px;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px #2EA043;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. MOTOR VE VERİ YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    model_path = os.path.join('models', 'epias_model_final.pkl')
    data_path = os.path.join('data_s', 'data_set_ex.xlsx')

    try:
        model_pkg = joblib.load(model_path)
        df = pd.read_excel(data_path)

        # Temizlik
        df.columns = [col.strip() for col in df.columns]
        df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')

        if 'Saat' in df.columns:
            if df['Saat'].dtype == 'O':
                df['Saat_Int'] = df['Saat'].astype(str).str.split(':').str[0].astype(int)
            else:
                df['Saat_Int'] = df['Saat']
        else:
            df['Saat_Int'] = df['Tarih'].dt.hour

        # Sayısal Temizlik
        cols = ['PTF (TL/MWH)', 'Yük Tahmin Planı (MWh)', 'Rüzgar', 'Güneş', 'Doğalgaz', 'İthal Kömür']
        for c in cols:
            if c in df.columns and df[c].dtype == 'O':
                df[c] = df[c].apply(
                    lambda x: float(str(x).replace('.', '').replace(',', '.')) if isinstance(x, str) else x)

        return model_pkg, df
    except Exception as e:
        return None, None


model_package, df_history = load_engine()

if not model_package:
    st.error("⚠️ Sistem Hatası: Model veya Veri Dosyası Bulunamadı!")
    st.stop()

model = model_package['model']
feature_list = model_package['features']


# -----------------------------------------------------------------------------
# 3. OTOMATİK VERİ ÇEKİCİ (HELPER)
# -----------------------------------------------------------------------------
def get_auto_data(tarih, saat):
    """Veri setinden otomatik değer çeker, yoksa ortalama döndürür."""
    mask = (df_history['Tarih'] == pd.to_datetime(tarih)) & (df_history['Saat_Int'] == saat)
    row = df_history[mask]

    if not row.empty:
        return row.iloc[0], "Gerçek Veri"
    else:
        m = pd.to_datetime(tarih).month
        mask_m = (df_history['Tarih'].dt.month == m) & (df_history['Saat_Int'] == saat)
        return df_history[mask_m].mean(numeric_only=True), "Simülasyon (Ortalama)"


def get_lag_prices(tarih, saat):
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
# 4. SIDEBAR - MANUEL KONTROL MERKEZİ
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Kontrol Paneli")

    # 1. Zaman Seçimi
    col_t1, col_t2 = st.columns(2)
    secilen_tarih = col_t1.date_input("Tarih", datetime.date(2025, 12, 27))
    secilen_saat = col_t2.selectbox("Saat", list(range(24)), index=9)

    # Otomatik Verileri Çek
    auto_row, veri_kaynagi = get_auto_data(secilen_tarih, secilen_saat)
    lag_24_auto, lag_168_auto = get_lag_prices(secilen_tarih, secilen_saat)

    st.caption(f"Veri Modu: {veri_kaynagi}")
    st.markdown("---")

    # 2. Manuel Girdi Alanları (Varsayılanları otomatikten alır)
    st.subheader("⚙️ Girdi Değişkenleri")


    def safe_val(key, default):
        return float(auto_row.get(key, default)) if auto_row is not None else default


    # Yük ve Yenilenebilir
    with st.expander("🏭 Üretim ve Yük", expanded=True):
        yuk = st.number_input("Yük Tahmini (MWh)", value=safe_val('Yük Tahmin Planı (MWh)', 35000))
        ruzgar = st.number_input("Rüzgar (MWh)", value=safe_val('Rüzgar', 3000))
        gunes = st.number_input("Güneş (MWh)", value=safe_val('Güneş', 0))
        dogalgaz = st.number_input("Doğalgaz (MWh)", value=safe_val('Doğalgaz', 8000))
        komur = st.number_input("İthal Kömür (MWh)", value=safe_val('İthal Kömür', 5000))

    # Fiyat Geçmişi
    with st.expander("💰 Fiyat Hafızası (Lag)", expanded=False):
        ptf_dun = st.number_input("Dünkü Fiyat (Lag 24)", value=float(lag_24_auto))
        ptf_hafta = st.number_input("Geçen Haftaki Fiyat", value=float(lag_168_auto))

    # Hesapla Butonu
    btn_predict = st.button("ANALİZİ GÜNCELLE 🚀")

# -----------------------------------------------------------------------------
# 5. ANA EKRAN VE GÖRSELLEŞTİRME
# -----------------------------------------------------------------------------
# Başlık
col_logo, col_head = st.columns([1, 10])
with col_logo:
    st.image(
        "foto/epias.jpg",
        width=50)
with col_head:
    st.title("EPİAŞ Pro Terminal")

# --- HESAPLAMA MOTORU ---
# (Her zaman çalışır, butona basılınca güncellenir veya ilk açılışta otomatiktir)

# Input DataFrame
input_df = pd.DataFrame([0], columns=['dummy'])
dt = pd.to_datetime(f"{secilen_tarih} {secilen_saat}:00:00")

# Tarihsel
input_df['Hour_Sin'] = np.sin(2 * np.pi * secilen_saat / 24)
input_df['Hour_Cos'] = np.cos(2 * np.pi * secilen_saat / 24)
input_df['Day_Sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
input_df['Day_Cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
input_df['Is_Weekend'] = 1 if dt.dayofweek >= 5 else 0
input_df['Is_Holiday'] = 1 if dt in holidays.TR() else 0

# Lag & Features (Manuel Girilen Değerler Kullanılır)
input_df['PTF_Lag_24'] = ptf_dun
input_df['PTF_Lag_168'] = ptf_hafta
rm = (ptf_dun + ptf_hafta) / 2
input_df['PTF_Roll_Mean_24'] = ptf_dun
input_df['PTF_Roll_Mean_168'] = rm
input_df['PTF_Roll_Std_24'] = 50
input_df['Relative_Price_Pos'] = (ptf_dun - rm) / (rm + 1)
input_df['Price_Momentum'] = ptf_dun - ptf_hafta

# Üretim Features (Manuel Girilenler)
ren = ruzgar + gunes
input_df['Total_Renewable_Lag24'] = ren
input_df['Net_Load'] = yuk - ren
therm = dogalgaz + komur
input_df['Total_Thermal_Lag24'] = therm
input_df['Thermal_Stress'] = therm / (yuk + 1)

# Shifted
input_df['Doğalgaz_Lag24'] = dogalgaz
input_df['Rüzgar_Lag24'] = ruzgar
input_df['Güneş_Lag24'] = gunes
input_df['İthal Kömür_Lag24'] = komur

# Eksikleri Tamamla
for c in feature_list:
    if c not in input_df.columns: input_df[c] = 0

# TAHMİN
pred = model.predict(input_df[feature_list])[0]
pred = max(0, pred)

# =============================================================================
# DASHBOARD GÖRÜNÜMÜ
# =============================================================================

# 1. KPI KARTLARI
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tahmini PTF", f"{pred:,.2f} ₺", f"{pred - ptf_dun:.1f} ₺ (Düne Fark)")
k2.metric("Sistem Yükü", f"{yuk:,.0f} MWh", "Talep Seviyesi", delta_color="off")
k3.metric("Net Yük (Termik Payı)", f"{yuk - ren:,.0f} MWh", f"Yenilenebilir: {ren:,.0f}")
k4.metric("Termik Stres", f"%{(therm / yuk) * 100:.1f}", "Risk Durumu", delta_color="inverse")

st.markdown("---")

# 2. GRAFİK BÖLÜMÜ (2 Satır, 2 Sütun)
r1_c1, r1_c2 = st.columns([1, 1])

# A. HIZ GÖSTERGESİ (GAUGE) - Fiyat Baskısı
with r1_c1:
    st.subheader("⚡ Fiyat Basınç Göstergesi")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 4000], 'tickwidth': 1},
            'bar': {'color': "#2EA043"},
            'bgcolor': "#0D1117",
            'steps': [
                {'range': [0, 1500], 'color': '#003B00'},
                {'range': [1500, 2500], 'color': '#3B3B00'},
                {'range': [2500, 4000], 'color': '#3B0000'}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': pred}}))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300,
                            margin=dict(t=30, b=0, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

# B. KIYASLAMA GRAFİĞİ (BAR CHART) - Dün vs Bugün
with r1_c2:
    st.subheader("📅 Fiyat Kıyaslaması")
    comp_df = pd.DataFrame({
        'Dönem': ['Geçen Hafta', 'Dün', 'BUGÜN (Tahmin)'],
        'Fiyat': [ptf_hafta, ptf_dun, pred],
        'Renk': ['#444', '#666', '#2EA043']
    })
    fig_bar = px.bar(comp_df, x='Dönem', y='Fiyat', color='Dönem',
                     text_auto='.0f', color_discrete_sequence=['#30363D', '#8B949E', '#238636'])
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={'color': "white"}, showlegend=False, height=300, margin=dict(t=30, b=0))
    fig_bar.update_traces(textfont_size=16, textposition='outside')
    st.plotly_chart(fig_bar, use_container_width=True)

r2_c1, r2_c2 = st.columns([1, 1])

# C. ENERJİ PASTASI (DONUT)
with r2_c1:
    st.subheader("🍰 Üretim Kaynakları")
    labels = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Kömür', 'Diğer']
    values = [ruzgar, gunes, dogalgaz, komur, max(0, yuk - (ruzgar + gunes + dogalgaz + komur))]

    fig_pie = px.pie(names=labels, values=values, hole=0.6,
                     color_discrete_sequence=px.colors.sequential.Tealgrn_r)
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"},
                          showlegend=True, height=300, margin=dict(t=30, b=0))
    # Ortaya Toplam Yükü Yazalım
    fig_pie.add_annotation(text=f"{yuk / 1000:.1f}k<br>MW", x=0.5, y=0.5, font_size=20, showarrow=False,
                           font_color="white")
    st.plotly_chart(fig_pie, use_container_width=True)

# D. YÜK DENGE ANALİZİ (AREA CHART)
with r2_c2:
    st.subheader("⚖️ Yük Dengesi")
    # Basit bir görselleştirme: Toplam Yük vs Net Yük
    balance_df = pd.DataFrame({
        'Metrik': ['Toplam Talep', 'Yenilenebilir', 'Termik Santrallere Kalan'],
        'Miktar': [yuk, ren, yuk - ren],
        'Tip': ['Talep', 'Arz', 'Sonuç']
    })
    # Waterfall tarzı bir bar chart
    fig_bal = go.Figure(go.Waterfall(
        name="20", orientation="v",
        measure=["relative", "relative", "total"],
        x=["Toplam Yük", "Yenilenebilir (-)", "Net Yük (=)"],
        textposition="outside",
        text=[f"{yuk:.0f}", f"-{ren:.0f}", f"{yuk - ren:.0f}"],
        y=[yuk, -ren, yuk - ren],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#2EA043"}},  # Yenilenebilir düşüşü yeşil
        increasing={"marker": {"color": "#DA3633"}},  # Yük artışı kırmızı
        totals={"marker": {"color": "#1F6FEB"}}  # Net yük mavi
    ))
    fig_bal.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={'color': "white"}, height=300, margin=dict(t=30, b=0))
    st.plotly_chart(fig_bal, use_container_width=True)

# F. DİPNOT (Yapay Zeka Yorumu)
st.info(f"💡 **AI Analisti:** Piyasa şu an %{(ren / yuk) * 100:.1f} oranında yenilenebilir enerji ile destekleniyor. "
        f"Fiyatlar {'düne göre artış' if pred > ptf_dun else 'düne göre düşüş'} trendinde. "
        f"{'Riskli Bölge! Gaz santralleri devrede.' if pred > 2500 else 'Piyasa stabil görünüyor.'}")