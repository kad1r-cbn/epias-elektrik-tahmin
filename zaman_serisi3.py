import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import jarque_bera
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols

# --- AYARLAR ---
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##################################################
# ADIM 1: VERİ TOPLAMA VE BİRLEŞTİRME
##################################################

# 1. Ham Veriyi Oku
df = pd.read_excel("data_s/güncel_set2.xlsx")


# 2. Sayısallaştırma (Nokta/Virgül temizliği)
def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    return float(x)


object_to_float = [col for col in df.columns if col not in ['Tarih', 'Saat']]
for col in object_to_float:
    df[col] = df[col].apply(clean_currency)

# 3. Zaman İndeksi Oluşturma
df['Tarih'] = pd.to_datetime(df['Tarih']).dt.normalize()
df['Saat_Sayısal'] = df['Saat'].astype(str).str[:2].astype(int)
df['Zaman'] = df['Tarih'] + pd.to_timedelta(df['Saat_Sayısal'], unit='h')
df.set_index('Zaman', inplace=True)

# 4. Dışsal Veriler (Dolar Kuru)
start, end = df['Tarih'].min(), df['Tarih'].max()
usd_data = yf.download('TRY=X', start=start, end=end + pd.Timedelta(days=5))['Close'].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']
usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize().dt.tz_localize(None)

df = df.reset_index().merge(usd_data, on='Tarih', how='left').set_index('Zaman')
df['Dolar_Kuru'] = df['Dolar_Kuru'].ffill().bfill()

# 5. Doğalgaz Fiyatları (Kademeli)
df['dogalgaz_fiyatlari_Mwh'] = 1100.0
df.loc[df['Tarih'] >= '2024-07-01', 'dogalgaz_fiyatlari_Mwh'] = 1127.82
df.loc[df['Tarih'] >= '2025-07-01', 'dogalgaz_fiyatlari_Mwh'] = 1409.77

##################################################
# ADIM 2: EDA VE TEMİZLİK
##################################################

df = df.interpolate(method='linear')
for col in df.columns:
    if col not in ['PTF (TL/MWH)', 'Tarih', 'Saat']:
        df.loc[df[col] < 0, col] = 0


##################################################
# ADIM 2.1: HETEROSKEDİSİTE (VARYANS) KONTROLÜ
##################################################

def check_and_transform(df):
    temp_df = df.copy().reset_index()
    temp_df['Time_Index'] = temp_df.index
    model = ols('Q("PTF (TL/MWH)") ~ Time_Index', data=temp_df).fit()

    # Breusch-Pagan Testi
    test = sms.het_breuschpagan(model.resid, model.model.exog)
    p_value = test[1]

    print(f"\n--- Heteroskedisite Testi (p-value: {p_value:.4e}) ---")

    if p_value < 0.05:
        print("Sonuç: Değişen Varyans VAR. Logaritmik Dönüşüm Uygulanıyor...")
        df['Target_PTF'] = np.log1p(df['PTF (TL/MWH)'])
        is_log = True
    else:
        print("Sonuç: Sabit Varyans. Orijinal veri korunuyor.")
        df['Target_PTF'] = df['PTF (TL/MWH)']
        is_log = False
    return df, is_log


df, log_applied = check_and_transform(df)

##################################################
# ADIM 3: İSTATİSTİKSEL TESTLER (ANALİZ VERİSİ ÜZERİNDEN)
##################################################

y_analysis = df['Target_PTF']


# A. Durağanlık (ADF)
def is_stationary(y):
    p_val = adfuller(y)[1]
    print(f"\n--- Durağanlık Testi (ADF) ---")
    print(f"p-value: {p_val:.4f} -> {'Durağan' if p_val < 0.05 else 'Durağan Değil'}")


# B. Normallik (Jarque-Bera)
def normality_test(y):
    stat, p_val = jarque_bera(y)
    print(f"\n--- Normallik Testi (JB) ---")
    print(f"p-value: {p_val:.4f} -> {'Normal' if p_val > 0.05 else 'Normal Değil'}")
    sns.histplot(y, kde=True)
    plt.title("Hedef Değişken Dağılımı")
    plt.show()


# C. Korelasyon ve D. VIF
def advanced_stats(df):
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['Target_PTF', 'PTF (TL/MWH)'], errors='ignore')

    # Korelasyon
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.show()

    # VIF
    vif_data = pd.DataFrame()
    X_vif = numeric_df.dropna()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    print("\n--- VIF Sonuçları ---")
    print(vif_data.sort_values(by="VIF", ascending=False))


is_stationary(y_analysis)
normality_test(y_analysis)
from statsmodels.stats.outliers_influence import variance_inflation_factor

advanced_stats(df)


##################################################
# ADIM 4: ZAMAN SERİSİ ANALİZİ (DECOMPOSITION)
##################################################

def ts_analysis(y):
    # Decomposition
    res = seasonal_decompose(y, model='additive', period=24)
    res.plot()
    plt.show()

    # ACF/PACF (Gecikme Tespiti İçin)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sm.graphics.tsa.plot_acf(y, lags=48, ax=ax[0])
    sm.graphics.tsa.plot_pacf(y, lags=48, ax=ax[1])
    plt.show()


ts_analysis(y_analysis)

print("\n>>> Analiz Katmanı Tamamlandı. Feature Engineering'e geçmeye hazırız.")


##################################################
# ADIM 4.1: GELİŞMİŞ GÖRSEL ANALİZ (Örüntü Avcılığı)
##################################################

def advanced_visual_analysis(df):
    # Geçici analiz sütunları oluşturalım
    analysis_df = df.copy()
    analysis_df['Saat'] = analysis_df.index.hour

    # Gün isimlerini Türkçe'ye çevirelim
    gunler_tr = {
        'Monday': 'Pazartesi', 'Tuesday': 'Salı', 'Wednesday': 'Çarşamba',
        'Thursday': 'Perşembe', 'Friday': 'Cuma', 'Saturday': 'Cumartesi', 'Sunday': 'Pazar'
    }
    analysis_df['Gun_Adi'] = analysis_df.index.day_name().map(gunler_tr)

    # Ay isimlerini Türkçe'ye çevirelim
    aylar_tr = {
        'January': 'Ocak', 'February': 'Şubat', 'March': 'Mart', 'April': 'Nisan',
        'May': 'Mayıs', 'June': 'Haziran', 'July': 'Temmuz', 'August': 'Ağustos',
        'September': 'Eylül', 'October': 'Ekim', 'November': 'Kasım', 'December': 'Aralık'
    }
    analysis_df['Ay_Adi'] = analysis_df.index.month_name().map(aylar_tr)

    # 1. Takvimsel Isı Haritası (Heatmap) - Saat vs Gün
    plt.figure(figsize=(15, 8))
    pivot_table = analysis_df.pivot_table(values='PTF (TL/MWH)',
                                          index='Saat',
                                          columns='Gun_Adi',
                                          aggfunc='mean')

    # Günleri sıralı listeleyelim (Pazartesi'den Pazar'a)
    sirali_gunler = ['Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi', 'Pazar']
    pivot_table = pivot_table.reindex(columns=sirali_gunler)

    sns.heatmap(pivot_table, cmap='YlOrRd', annot=False)
    plt.title("Saatlik ve Günlük PTF Ortalamaları (Isı Haritası - TL/MWh)", fontsize=14)
    plt.xlabel("Günler")
    plt.ylabel("Saat")
    plt.show()

    # 2. Mevsimsel Kutu Grafikleri (Boxplots) - Saatlik Oynaklık
    plt.figure(figsize=(15, 6))
    sns.boxplot(x='Saat', y='PTF (TL/MWH)', data=analysis_df, palette="husl")
    plt.title("Saat Bazlı Fiyat Dağılımı ve Oynaklık (Boxplot)", fontsize=14)
    plt.xlabel("Günün Saatleri")
    plt.ylabel("Fiyat (TL/MWh)")
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 3. Aylık Bazda Oynaklık
    plt.figure(figsize=(15, 6))
    sirali_aylar = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                    'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']

    sns.boxplot(x='Ay_Adi', y='PTF (TL/MWH)', data=analysis_df,
                order=[a for a in sirali_aylar if a in analysis_df['Ay_Adi'].unique()],
                palette="Set3")
    plt.title("Aylık Bazlı Fiyat Dağılımı (Mevsimsel Değişimler)", fontsize=14)
    plt.xlabel("Aylar")
    plt.ylabel("Fiyat (TL/MWh)")
    plt.xticks(rotation=45)
    plt.show()


# Fonksiyonu çalıştıralım
advanced_visual_analysis(df)



#bu kısımda acf ve pacf grafiklerini yapıp bunlara bakarak parametreleri belirlememiz lazım
#akaike parametresi ile de otomatik olarak auto_arima ile de bulabiliriz en düşük gelen değerlerimizi




##################################################
# ADIM 5: FEATURE ENGINEERING (ÖZELLİK ÜRETİMİ)
##################################################

print("\n>>> Analiz bulgularına göre yeni özellikler (features) üretiliyor...")

# Isı haritası ve boxplot analizinden elde edilen çıktılarla modelimizi güçlendiriyoruz
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# Boxplot'ta gördüğümüz yüksek fiyatlı (Peak) saatleri işaretleyelim
# Genelde sabah 08-11 ve akşam 17-21 arası Türkiye elektrik piyasasında peak'tir.
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if x in [8, 9, 10, 11, 17, 18, 19, 20] else 0)

# ACF/PACF'den gelen gecikme (Lag) özellikleri
# Hocaya: "Fiyatların dünkü ve geçen haftaki aynı saatle korelasyonu yüksek."
df['PTF_Lag_24'] = df['Target_PTF'].shift(24)    # 1 gün önce
df['PTF_Lag_168'] = df['Target_PTF'].shift(168)  # 1 hafta önce

# Hareketli Ortalama (Trend takibi için)
df['PTF_Rolling_Mean_24'] = df['Target_PTF'].shift(1).rolling(window=24).mean()

# Laglardan dolayı oluşan NaN satırları silelim
df.dropna(inplace=True)

print(">>> Özellik üretimi tamamlandı. Model eğitimine hazırız.")


df.head()









