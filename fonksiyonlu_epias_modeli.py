# =============================================================================
# 0. KÜTÜPHANE VE AYARLAR (IMPORTS)
# =============================================================================

# 1. Standart Python ve Uyarılar
import warnings
# Gereksiz uyarıları kapat (Temiz çıktı için)
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 2. Veri İşleme ve Matematik (Data Manipulation)
import numpy as np
import pandas as pd
import holidays

# 3. Görselleştirme (Visualization)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# 4. İstatistik ve Zaman Serisi Analizi (Statistics)
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson  # Durbin-Watson eklendi
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 5. Dış Veri Kaynakları (External Data)
import yfinance as yf

# 6. Makine Öğrenmesi ve Metrikler (Machine Learning)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 7. Açıklanabilir Yapay Zeka (XAI)
import shap

# Görselleştirme Ayarları (Opsiyonel ama önerilir)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

print("✅ Tüm kütüphaneler başarıyla yüklendi ve ayarlandı.")


# ---------------------------
# AYARLAR
# ---------------------------

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ---------------------------
# VERİ OKUMA
# ---------------------------
"""df_final = pd.read_csv("data_s/data_set.csv")"""
df_final = pd.read_excel("data_s/data_set_ex.xlsx")
df_final_c = df_final.copy()
# ---------------------------
# DEĞİŞKEN TİPİ DÜZELTME
# --------------------------

def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    return float(x)

object_to_float = [col for col in df_final.columns if col not in ['Tarih', 'Saat', 'Zaman']]
for col in object_to_float:
    df_final[col] = df_final[col].apply(clean_currency)

df_final['Tarih'] = pd.to_datetime(df_final['Tarih'], format='%d.%m.%Y').dt.normalize()
df_final['Zaman'] = pd.to_datetime(df_final['Tarih'].astype(str) + ' ' + df_final['Saat'].astype(str))



df_final['Tarih'] = pd.to_datetime(df_final['Tarih'], format='%d.%m.%Y').dt.normalize()
df_final['Saat'] = pd.to_datetime(df_final['Saat']).dt.time
df_final.head()
df_final.info()


# ---------------------------
# DOLAR VE BOTAŞ EKLEME
# ---------------------------

# ---------------------------
# DOLAR KURU (YAHOO)
# ---------------------------
start_date = df_final['Tarih'].min()
end_date = df_final['Tarih'].max()
usd_data = yf.download('TRY=X', start=start_date, end=end_date + pd.Timedelta(days=5))
usd_data = usd_data[['Close']].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']
usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize()
usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)
# Eksik günleri doldur
all_dates = pd.DataFrame({'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')})
all_dates['Tarih'] = all_dates['Tarih'].dt.normalize()
usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')
usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()
# Ana veriye ekle
df_final = pd.merge(df_final, usd_data, on='Tarih', how='left')

# ---------------------------
# BOTAŞ DOĞALGAZ VERİSİ
# ---------------------------
#
df_final['dogalgaz_fiyatlari_Mwh'] = 1692.00
df_final.loc[df_final['Tarih'] >= '2024-07-02', 'dogalgaz_fiyatlari_Mwh'] = 1127.82
df_final.loc[df_final['Tarih'] >= '2025-07-01', 'dogalgaz_fiyatlari_Mwh'] = 1409.77

df_final.describe().T
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#   ADIM 0                --EDA--
# =============================================================================
# =============================================================================
# YARDIMCI FONKSİYONLAR (ALT PARÇALAR)
# =============================================================================

def data_summary(dataframe, head=5):
    """Veri setinin genel özetini basar."""
    print("\n##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Type #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA Check #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0.05, 0.50, 0.95, 0.99]).T)


def clean_negative_values(dataframe, col_name='Güneş'):
    """Negatif değerleri temizler ve 0'a eşitler."""
    if col_name in dataframe.columns:
        print(f"\n--- '{col_name}' Değişkeni Temizliği ---")
        print(f"Negatif Sayısı: {(dataframe[col_name] < 0).sum()}")
        dataframe[col_name] = dataframe[col_name].clip(lower=0)
        print(f"Düzeltme Sonrası Negatif Sayısı: {(dataframe[col_name] < 0).sum()}")
    return dataframe


def degisken_analiz(dataframe, cat_th=2, car_th=20):
    """Kategorik ve Numerik değişkenleri ayırır."""
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"\n--- Değişken Analizi ---")
    print(f"Gözlem Sayısı: {dataframe.shape[0]}")
    print(f"Değişken Sayısı: {dataframe.shape[1]}")
    print(f'Kategorik Değişkenler: {len(cat_cols)}')
    print(f'Numerik Değişkenler: {len(num_cols)}')
    print(f'Kardinalitesi Yüksek Kategorikler: {len(cat_but_car)}')
    print(f'Numerik Görünümlü Kategorikler: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def numeric_summary(dataframe, numerical_col, plot=False):
    """Numerik değişkenlerin istatistiklerini ve histogramını çizer."""
    quantiles = [0.05, 0.25, 0.50,0.75,0.95]
    print(f"\n###### {numerical_col} Özeti ######")
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


def target_summary_with_numeric(dataframe, target, numerical_col):
    """Hedef değişkene göre numerik değişkenlerin ortalamasını alır."""
    print(f"\n--- {target} Kırılımında {numerical_col} Ortalaması ---")
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))


def plot_correlation_matrix(dataframe, num_cols):
    """Korelasyon matrisini çizer."""
    print("\n--- Korelasyon Matrisi Çiziliyor ---")
    if len(num_cols) > 1:
        corr = dataframe[num_cols].corr()
        f, ax = plt.subplots(figsize=[18, 13])
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="magma")
        ax.set_title("Correlation Matrix", fontsize=20)
        plt.show(block=True)
    else:
        print("Yeterli sayısal değişken yok.")


def check_physical_integrity(df):
    """Fiziksel mantık kontrollerini yapar."""
    print("\n🕵️‍♂️ Fiziksel Tutarlılık Kontrolü Yapılıyor...")

    # 1. Negatif Üretim Kontrolü
    prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit']
    existing_cols = [c for c in prod_cols if c in df.columns]

    for col in existing_cols:
        negatives = df[df[col] < 0]
        if len(negatives) > 0:
            print(f"⚠️ UYARI: {col} sütununda {len(negatives)} adet negatif değer var! 0'a eşitleniyor.")
            df.loc[df[col] < 0, col] = 0
        else:
            print(f"✅ {col}: Temiz (Negatif yok).")

    # 2. PTF Kontrolü
    MAX_PRICE_LIMIT = 6000
    MIN_PRICE_LIMIT = 0
    if 'PTF (TL/MWH)' in df.columns:
        errors = df[(df['PTF (TL/MWH)'] > MAX_PRICE_LIMIT) | (df['PTF (TL/MWH)'] < MIN_PRICE_LIMIT)]
        if len(errors) > 0:
            print(f"🚨 KRİTİK: PTF sütununda {len(errors)} adet mantıksız değer var!")
        else:
            print("✅ PTF: Mantıksız uç değer (Error) görünmüyor.")

    return df


def plot_all_boxplots(df):
    """Gruplandırılmış Boxplotları çizer."""
    print("\n--- Boxplot Analizi Çiziliyor ---")
    sns.set_theme(style="whitegrid")

    # Grupları mevcut sütunlara göre filtrele
    price_cols = [c for c in ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh'] if c in df.columns]
    large_scale_cols = [c for c in ['Yük Tahmin Planı (MWh)', 'Doğalgaz', 'Barajlı', 'İthal Kömür'] if c in df.columns]
    renewable_cols = [c for c in ['Rüzgar', 'Güneş', 'Akarsu', 'Linyit', 'Jeotermal', 'Biyokütle', 'Fuel Oil'] if
                      c in df.columns]

    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    if price_cols:
        sns.boxplot(data=df[price_cols], ax=axes[0], palette="Set2")
        axes[0].set_title('Grup 1: Fiyat Bazlı Değişkenler')

    if large_scale_cols:
        sns.boxplot(data=df[large_scale_cols], ax=axes[1], palette="Set1")
        axes[1].set_title('Grup 2: Yük ve Büyük Ölçekli Üretimler')

    if renewable_cols:
        sns.boxplot(data=df[renewable_cols], ax=axes[2], palette="Pastel1")
        axes[2].set_title('Grup 3: Yenilenebilir Enerji ve Diğer Üretimler')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# ANA YÖNETİCİ FONKSİYON (MASTER FUNCTION)
# =============================================================================

def run_full_eda(dataframe, target_col="PTF (TL/MWH)", plot_hists=False):
    """
    Tüm EDA sürecini tek seferde çalıştırır.

    Parametreler:
    dataframe: Analiz edilecek pandas DataFrame
    target_col: Hedef değişken ismi (Örn: 'PTF (TL/MWH)')
    plot_hists: Numerik değişkenlerin histogramlarını çizip çizmeyeceği (True/False)
    """
    df = dataframe.copy()

    # 1. Genel Bakış
    data_summary(df)

    # 2. Temizlik (Güneş vb.)
    df = clean_negative_values(df, col_name='Güneş')

    # 3. Değişkenlerin Ayrıştırılması
    cat_cols, num_cols, cat_but_car = degisken_analiz(df)

    # 4. Numerik Değişken Analizi
    print("\n--- NUMERİK DEĞİŞKENLERİN ANALİZİ ---")
    for col in num_cols:
        numeric_summary(df, numerical_col=col, plot=plot_hists)

    # 5. Target Analizi
    if target_col in df.columns:
        print(f"\n--- HEDEF DEĞİŞKEN ({target_col}) ANALİZİ ---")
        for col in num_cols:
            if col != target_col:
                # Target numerik olduğu için scatter veya corr bakmak daha mantıklı olsa da
                # mevcut koddaki yapıyı korumak adına groupby ile özet geçiyoruz (eğer kategorik target olsaydı)
                # Ancak sürekli (continuous) target için korelasyon daha iyidir.
                pass

                # 6. Korelasyon Matrisi
    plot_correlation_matrix(df, num_cols)

    # 7. Fiziksel Tutarlılık
    df = check_physical_integrity(df)

    # 8. Boxplot Analizi
    plot_all_boxplots(df)

    print("\n✅ EDA Süreci Tamamlandı.")
    return df

# =============================================================================
# KULLANIM
# =============================================================================

# Tek satırda çalıştırmak için

df_final = run_full_eda(df_final, target_col="PTF (TL/MWH)", plot_hists=True)
#////////////////////////////////////////////////////////////////////////////////////










# =============================================================================
# ADIM 1 İSTATİSTİK FONKSİYONLARI
# =============================================================================

def check_normality(dataframe, target_col):
    """
    Hedef değişkenin Normal Dağılıma uyup uymadığını test eder (K-S Testi, Histogram, Q-Q Plot).
    """
    print(f"\n" + "=" * 50)
    print(f"📊 NORMALLİK TESTİ: {target_col}")
    print("=" * 50)

    # Veriyi temizle
    data_clean = dataframe[target_col].dropna()

    # 1. Kolmogorov-Smirnov Testi
    # Veriyi standardize edip teste sokuyoruz
    ks_stat, p_value_ks = stats.kstest((data_clean - data_clean.mean()) / data_clean.std(), 'norm')
    print(f"K-S Testi İstatistiği: {ks_stat:.4f}")
    print(f"K-S Testi p-değeri:    {p_value_ks:.4f}")

    if p_value_ks < 0.05:
        print("-> Sonuç: Veri Normal Dağılıma UYMUYOR (H0 Red).")
    else:
        print("-> Sonuç: Veri Normal Dağılıma UYUYOR (H0 Reddedilemez).")

    print(f"Çarpıklık (Skewness):  {data_clean.skew():.4f}")
    print(f"Basıklık (Kurtosis):   {data_clean.kurt():.4f}")

    # 2. Histogram ve Teorik Normal Eğri
    plt.figure(figsize=(10, 6))
    sns.histplot(data_clean, kde=True, stat="density", color='skyblue', label='Gerçek Dağılım')

    mu, std = data_clean.mean(), data_clean.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2, label='Teorik Normal Dağılım')
    plt.title(f'{target_col} Dağılımı vs Teorik Normal Dağılım')
    plt.legend()
    plt.show()

    # 3. Q-Q Plot
    fig = sm.qqplot(data_clean, line='s')
    plt.title(f'{target_col} İçin Q-Q Plot')
    plt.show()


def check_stationarity(dataframe, target_col, plot_cols=None):
    """
    Tüm sayısal değişkenler için ADF (Augmented Dickey-Fuller) Durağanlık testi yapar.
    """
    print(f"\n" + "=" * 50)
    print("📈 DURAĞANLIK (STATIONARITY) TESTİ - ADF")
    print("=" * 50)

    # Sadece sayısal sütunları al (Tarih ve zaman hariç)
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['Tarih', 'Zaman', 'Saat']]

    adf_results = []

    for col in numeric_cols:
        series = dataframe[col].dropna()
        # Sabit varyans varsa test hatası almamak için kontrol
        if series.nunique() <= 1:
            continue

        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        is_stationary = "✅ Evet" if p_value <= 0.05 else "❌ Hayır"

        adf_results.append({
            'Değişken': col,
            'ADF Stat': round(result[0], 4),
            'p-değeri': round(p_value, 4),
            'Durağan mı?': is_stationary
        })

        # Hedef değişken için detaylı yazdır
        if col == target_col:
            print(f"--- {target_col} İçin ADF Detayı ---")
            print(f"ADF İstatistiği: {result[0]:.4f}")
            print(f"p-değeri: {result[1]:.4f}")
            print("Kritik Değerler:", result[4])
            print(f"SONUÇ: Seri {'DURAĞANDIR' if p_value <= 0.05 else 'Durağan DEĞİLDİR (Trend Var)'}.\n")

    # Sonuç Tablosu
    adf_df = pd.DataFrame(adf_results)
    print("--- Tüm Değişkenler İçin Özet Tablo ---")
    print(adf_df)

    # Seçili Değişkenlerin Zaman Serisi Grafiği
    if plot_cols:
        valid_cols = [c for c in plot_cols if c in dataframe.columns]
        if valid_cols:
            fig, axes = plt.subplots(len(valid_cols), 1, figsize=(12, 4 * len(valid_cols)))
            if len(valid_cols) == 1: axes = [axes]  # Tekli durumda döngü hatası olmasın

            for i, col in enumerate(valid_cols):
                axes[i].plot(dataframe.index, dataframe[col], color='tab:blue')
                axes[i].set_title(f'{col} - Zaman Serisi Grafiği')
                axes[i].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


def analyze_volatility(dataframe):
    """
    Belirli gruplar için hareketli standart sapma (Volatilite) analizi yapar.
    Hem grafik çizer hem de istatistiksel tabloyu basar.
    """
    print(f"\n" + "=" * 50)
    print("📉 VOLATİLİTE ANALİZİ (24 Saatlik Rolling Std)")
    print("=" * 50)

    groups = {
        "Fiyat ve Kur": ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh'],
        "Fosil Yakıtlar": ['Doğalgaz', 'Linyit', 'İthal Kömür'],
        "Yenilenebilir": ['Akarsu', 'Rüzgar', 'Güneş']
    }

    for title, cols in groups.items():
        valid_cols = [c for c in cols if c in dataframe.columns]

        if valid_cols:
            # Hesaplama
            vol_data = dataframe[valid_cols].rolling(window=24).std()

            # --- SAYISAL ÇIKTI KISMI (YENİ EKLENDİ) ---
            print(f"\n📊 GRUP: {title} - Volatilite İstatistikleri")
            print("-" * 45)
            # Ortalama, Max ve Min oynaklığı gösteren tablo
            stats_summary = vol_data.describe().T[['mean', 'std', 'min', 'max']]
            print(stats_summary)

            # Grafik Çizimi
            vol_data.plot(figsize=(12, 5), title=f"{title} Volatilitesi")
            plt.ylabel("Standart Sapma (24s)")
            plt.grid(True, alpha=0.3)
            plt.show()


def analyze_correlation(dataframe, target_col):
    """
    Spearman korelasyonu hesaplar.
    Hem sayısal listeyi basar hem de Heatmap çizer.
    """
    print(f"\n" + "=" * 50)
    print("🔗 KORELASYON ANALİZİ (Spearman)")
    print("=" * 50)

    num_cols = dataframe.select_dtypes(include=[np.number]).columns

    if len(num_cols) < 2:
        return

    # Hesaplama
    corr_matrix = dataframe[num_cols].corr(method='spearman')

    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[[target_col]].sort_values(by=target_col, ascending=False)

        # --- SAYISAL ÇIKTI KISMI (YENİ EKLENDİ) ---
        print(f"\n🔢 {target_col} ile Korelasyon Katsayıları (Sıralı Liste):")
        print("-" * 50)
        # Tabloyu daha okunaklı bas
        print(target_corr)
        print("-" * 50)

        # Grafik Çizimi
        plt.figure(figsize=(6, 10))
        sns.heatmap(target_corr, annot=True, cmap='RdYlGn', fmt=".2f", center=0)
        plt.title(f"{target_col} ile Spearman Korelasyonu")
        plt.show()
    else:
        print(f"Hata: {target_col} korelasyon matrisinde bulunamadı.")


def analyze_vif(dataframe, target_col, drop_list=None):
    """
    Çoklu Bağlantı (Multicollinearity) Analizi - VIF
    4 Aşamalı Test Yapar: Raw, Reduced, Scaled, Differenced
    """
    print(f"\n" + "=" * 50)
    print("🧩 ÇOKLU BAĞLANTI (VIF) ANALİZİ")
    print("=" * 50)

    # Target hariç bağımsız değişkenler
    X = dataframe.drop([target_col], axis=1).select_dtypes(include=[np.number])
    # NaN temizliği
    X = X.dropna()

    def calc_vif(data, label):
        """VIF hesaplayan yardımcı iç fonksiyon"""
        if data.shape[1] == 0: return
        vif_df = pd.DataFrame()
        vif_df["Değişken"] = data.columns
        vif_df["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        print(f"\n--- {label} ---")
        print(vif_df.sort_values(by="VIF", ascending=False).head(10))  # İlk 10'u göster
        return vif_df

    # 1. Ham Veri VIF
    calc_vif(X, "1. Ham Veri VIF Sonuçları")

    # 2. Belirli Değişkenleri Çıkararak VIF
    if drop_list:
        valid_drop = [c for c in drop_list if c in X.columns]
        X_reduced = X.drop(columns=valid_drop)
        calc_vif(X_reduced, "2. Gereksiz Değişkenler Atıldıktan Sonra VIF")

    # 3. Ölçeklenmiş (Scaled) VIF
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    calc_vif(X_scaled, "3. StandardScaler Sonrası VIF")

    # 4. Farkı Alınmış (Differenced) VIF
    X_diff = X.diff().dropna()
    calc_vif(X_diff, "4. Fark Alma (Differencing) Sonrası VIF")


# =============================================================================
# ANA YÖNETİCİ FONKSİYON
# =============================================================================

def run_statistical_tests(dataframe, target_col="PTF (TL/MWH)"):
    """
    Tüm istatistiksel testleri sırayla çalıştırır.
    """
    df = dataframe.copy()

    # 1. Normallik
    check_normality(df, target_col)

    # 2. Durağanlık (ADF)
    # Grafiği çizilecek kritik değişkenler (varsa)
    critical_cols = ['Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh', 'Akarsu', 'Jeotermal']
    check_stationarity(df, target_col, plot_cols=critical_cols)

    # 3. Volatilite
    analyze_volatility(df)

    # 4. Korelasyon
    analyze_correlation(df, target_col)

    # 5. VIF (Çoklu Bağlantı)
    # Çıkarılması düşünülen yüksek VIF'li değişkenler listesi
    cols_to_drop = ['Biyokütle', 'Jeotermal', 'Akarsu']
    analyze_vif(df, target_col, drop_list=cols_to_drop)

    print("\n✅ Tüm İstatistiksel Testler Tamamlandı.")

# =============================================================================
# KULLANIM
# =============================================================================

# Tek satırda çalıştırmak için:
run_statistical_tests(df_final, target_col="PTF (TL/MWH)")
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#      ADIM 2         --TİME SERİES (ZAMAN SERİSİ)--
# =============================================================================

def run_time_series_analysis(dataframe, target_col="PTF (TL/MWH)", feature_col="Rüzgar"):
    """
    Zaman Serisi Analizi Paketi (Sayısal Raporlu Versiyon):
    Hem grafik çizer hem de konsola istatistiksel özet basar.
    """
    print(f"\n" + "=" * 50)
    print("⏳ ZAMAN SERİSİ ANALİZİ BAŞLIYOR (Sayısal Raporlu)")
    print("=" * 50)

    df = dataframe.copy()
    series = df[target_col].dropna()

    # =========================================================================
    # 1. ZAMAN BİLGİSİ HAZIRLIĞI
    # =========================================================================
    # Saat, Gün, Ay bilgilerini çıkar
    if 'Saat' in df.columns:
        try:
            df['Hour_Viz'] = df['Saat'].astype(str).str.split(':').str[0].astype(int)
        except:
            df['Hour_Viz'] = df['Saat'].astype(int)
    elif 'Tarih' in df.columns:
        df['Hour_Viz'] = df['Tarih'].dt.hour
    else:
        df['Hour_Viz'] = df.index % 24

    if 'Tarih' in df.columns:
        df['Day_of_Week'] = df['Tarih'].dt.dayofweek
        df['Month'] = df['Tarih'].dt.month
    else:
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month

    # =========================================================================
    # 2. MEVSİMSELLİK AYRIŞTIRMA (DECOMPOSITION)
    # =========================================================================
    print("\n1. Mevsimsel Ayrıştırma (Decomposition)")
    try:
        res = seasonal_decompose(series, model='additive', period=24)

        # --- SAYISAL ÇIKTI ---
        print(f"   Ortalama Trend Değeri: {res.trend.mean():.2f}")
        print(f"   Mevsimsellik Etkisi (Max): {res.seasonal.max():.2f}")
        print(f"   Mevsimsellik Etkisi (Min): {res.seasonal.min():.2f}")

        # Grafik
        plt.rcParams['figure.figsize'] = (14, 10)
        res.plot()
        plt.suptitle(f'{target_col} - 24 Saatlik Ayrıştırma', fontsize=16, y=1.02)
        plt.show()
    except Exception as e:
        print(f"❌ Decomposition hatası: {e}")

    # =========================================================================
    # 3. ACF ve PACF (OTOKORELASYON)
    # =========================================================================
    print("\n2. Otokorelasyon Analizi")
    print("   (Grafikler oluşturuluyor... ACF: Hafıza, PACF: Doğrudan Etki)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    # 48 Saat
    plot_acf(series, lags=48, ax=axes[0, 0], title='ACF (48 Saat)')
    plot_pacf(series, lags=48, ax=axes[0, 1], method='yw', title='PACF (48 Saat)')
    # 168 Saat
    plot_acf(series, lags=168, ax=axes[1, 0], title='ACF (1 Hafta)')
    plot_pacf(series, lags=168, ax=axes[1, 1], method='yw', title='PACF (1 Hafta)')
    plt.tight_layout()
    plt.show()

    # =========================================================================
    # 4. ÇAPRAZ KORELASYON (FEATURE vs TARGET)
    # =========================================================================
    if feature_col in df.columns:
        print(f"\n3. Çapraz Korelasyon: {feature_col} vs {target_col}")

        feat_series = df[feature_col].dropna()
        min_len = min(len(series), len(feat_series))
        s1 = series.iloc[:min_len]
        s2 = feat_series.iloc[:min_len]

        cross_corr = [s1.corr(s2.shift(lag)) for lag in range(25)]

        # --- SAYISAL ÇIKTI (ÖNEMLİ) ---
        print("-" * 40)
        print(f"   Gecikme (Lag) | Korelasyon Katsayısı")
        print("-" * 40)
        for i, val in enumerate(cross_corr[:6]):  # İlk 5 saati bas
            print(f"   Lag {i} (Saat)  | {val:.4f}")

        # En güçlü ilişkiyi bul
        max_idx = np.argmax(np.abs(cross_corr))
        print("-" * 40)
        print(f"👉 EN GÜÇLÜ İLİŞKİ: {max_idx}. Saatte (Corr: {cross_corr[max_idx]:.4f})")
        print("-" * 40)

        # Grafik
        plt.figure(figsize=(10, 5))
        plt.bar(range(25), cross_corr, color='teal')
        plt.title(f'{feature_col} ve {target_col} Gecikmeli İlişki')
        plt.xlabel('Gecikme (Saat)')
        plt.show()
    else:
        print(f"⚠️ '{feature_col}' bulunamadı.")

    # =========================================================================
    # 5. HAREKETLİ ORTALAMA VE OYNAKLIK
    # =========================================================================
    print("\n4. Trend ve Volatilite İstatistikleri")

    rolling_mean = series.rolling(window=24).mean()
    rolling_std = series.rolling(window=24).std()

    # --- SAYISAL ÇIKTI ---
    print(f"   Genel Ortalama Fiyat: {series.mean():.2f}")
    print(f"   Ortalama Volatilite (Std): {rolling_std.mean():.2f}")
    print(f"   Maksimum Volatilite: {rolling_std.max():.2f}")

    # Grafik
    plt.figure(figsize=(14, 6))
    plt.plot(series, label='Gerçek', alpha=0.3, color='gray')
    plt.plot(rolling_mean, label='Hareketli Ort.', color='red')
    plt.plot(rolling_std, label='Hareketli Std.', color='blue', linestyle='--')
    plt.title('Trend ve Volatilite')
    plt.legend()
    plt.show()

    # =========================================================================
    # 6. ISI HARİTASI VE DETAYLI TABLO
    # =========================================================================
    print("\n5. Saatlik Fiyat Matrisi (Pivot Tablo)")

    if 'Hour_Viz' in df.columns and 'Day_of_Week' in df.columns:
        pivot_table = df.pivot_table(values=target_col, index='Hour_Viz', columns='Day_of_Week', aggfunc='mean')

        # --- SAYISAL ÇIKTI (TABLOYU BAS) ---
        # Okunabilirlik için sütun isimlerini değiştir
        gunler = {0: 'Pzt', 1: 'Sal', 2: 'Çar', 3: 'Per', 4: 'Cum', 5: 'Cmt', 6: 'Paz'}
        pivot_print = pivot_table.rename(columns=gunler)

        print("\n   Saatlere ve Günlere Göre Ortalama PTF:")
        print(pivot_print.round(2))  # Virgülden sonra 2 hane ile tabloyu bas

        # Grafik
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=False)
        plt.title('PTF Ortalaması: Saat vs Gün')
        plt.show()

        # Boxplot
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        sns.boxplot(x='Hour_Viz', y=target_col, data=df, ax=axes[0], palette="viridis")
        axes[0].set_title('Saat Bazlı Dağılım')

        if 'Month' in df.columns:
            sns.boxplot(x='Month', y=target_col, data=df, ax=axes[1], palette="magma")
            axes[1].set_title('Aylık Dağılım')
        plt.show()
    else:
        print("⚠️ Saat verisi eksik.")

    print("\n✅ Analiz Tamamlandı.")


# Çalıştırma
run_time_series_analysis(df_final, target_col="PTF (TL/MWH)", feature_col="Rüzgar")
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#     ADIM 3            --FUTURE ENGENEERİNG--
# =============================================================================
def run_feature_engineering(dataframe):
    """
    Ham veri setini alır, 'Sniper' özellikleri ekler, Sızıntı (Leakage) kontrolü yapar
    ve modele hazır hale getirir.
    """
    print(f"\n" + "=" * 50)
    print("🛠️ FEATURE ENGINEERING (ÖZELLİK MÜHENDİSLİĞİ) BAŞLIYOR")
    print("=" * 50)

    # Orijinal veriyi bozmamak için kopya al
    df = dataframe.copy()

    # -------------------------------------------------------------------------
    # 1. TATİL VE ZAMAN DEĞİŞKENLERİ
    # -------------------------------------------------------------------------
    print("📅 Takvim ve Tatil Verileri İşleniyor...")

    # Tarih formatı kontrolü
    if 'Tarih' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'Tarih'}, inplace=True)
        else:
            print("❌ HATA: 'Tarih' sütunu bulunamadı!")
            return None

    df['Tarih'] = pd.to_datetime(df['Tarih'])

    # Tatil Günleri (Türkiye)
    try:
        tr_holidays = holidays.TR(years=[2023, 2024, 2025])
        df['Is_Holiday'] = df['Tarih'].apply(lambda x: 1 if x in tr_holidays else 0)
    except Exception as e:
        print(f"⚠️ Tatil verisi çekilemedi: {e}")
        df['Is_Holiday'] = 0

    # Hafta Sonu ve Günler
    df['Day_of_Week'] = df['Tarih'].dt.dayofweek
    df['Month'] = df['Tarih'].dt.month
    df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)

    # Saat Dönüşümü (Döngüsel Özellikler)
    # Saat sütunu yoksa Tarih'ten çek, varsa işle
    if 'Saat' in df.columns:
        if df['Saat'].dtype == 'O':  # Object/String ise
            df['Saat_Int'] = df['Saat'].astype(str).str.split(':').str[0].astype(int)
        else:
            df['Saat_Int'] = df['Saat'].astype(int)
    else:
        df['Saat_Int'] = df['Tarih'].dt.hour

    # Sinüs/Kosinüs Dönüşümü (Saatin 23 ile 00 arasındaki yakınlığını modele öğretmek için)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Saat_Int'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Saat_Int'] / 24)
    df['Day_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

    # -------------------------------------------------------------------------
    # 2. SHIFT (LAG) OPERASYONU (GELECEĞİ GÖRMEYİ ENGELLEME)
    # -------------------------------------------------------------------------
    # Bu değişkenler gerçekleşen verilerdir. Yarını tahmin ederken bugünün değerini bilemeyiz.
    # O yüzden 24 saat öncesini (dünü) kullanıyoruz.
    future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit',
                   'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal', 'Biyokütle']

    # Sadece veri setinde var olanları seç
    cols_to_shift = [c for c in future_cols if c in df.columns]

    print(f"⏳ Shift İşlemi: {len(cols_to_shift)} üretim değişkeni ötelenecek...")

    for col in cols_to_shift:
        # Zaten Lag24 yapılmış mı kontrol et (Çift çalışmayı önle)
        if f'{col}_Lag24' not in df.columns:
            df[f'{col}_Lag24'] = df[col].shift(24)
            # Orijinal sütunu sil (Model kopya çekmesin)
            df.drop(columns=[col], inplace=True)

    # Trend Analizi (Fark Alma)
    # Örneğin: Doğalgaz düne göre arttı mı azaldı mı?
    trend_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Dolar_Kuru']
    for col in trend_cols:
        if col in df.columns:
            df[f'{col}_Diff'] = df[col].diff()

    # -------------------------------------------------------------------------
    # 3. FİYAT HAFIZASI (TARGET LAGS) - KRİTİK BÖLÜM
    # -------------------------------------------------------------------------
    target_col = 'PTF (TL/MWH)'

    if target_col in df.columns:
        # Dün bu saatte fiyat neydi?
        df['PTF_Lag_24'] = df[target_col].shift(24)
        # Geçen hafta bu saatte fiyat neydi?
        df['PTF_Lag_168'] = df[target_col].shift(168)

        # --- GÜVENLİ ROLLING (SIZINTI ÖNLEYİCİ) ---
        # Hareketli ortalamayı 'PTF' üzerinden DEĞİL, 'PTF_Lag_24' üzerinden alıyoruz.
        # Böylece bugünün verisi hesabın içine karışmıyor.
        df['PTF_Roll_Mean_24'] = df['PTF_Lag_24'].rolling(24).mean()
        df['PTF_Roll_Std_24'] = df['PTF_Lag_24'].rolling(24).std()
        df['PTF_Roll_Mean_168'] = df['PTF_Lag_24'].rolling(168).mean()
    else:
        print("❌ HATA: Hedef değişken (PTF) bulunamadı!")
        return None

    # -------------------------------------------------------------------------
    # 4. SNIPER ÖZELLİKLER (AKILLI RASYOLAR)
    # -------------------------------------------------------------------------
    print("🎯 Sniper Değişkenler (Rasyolar) Oluşturuluyor...")

    # A. Relative Price Position (Fiyatın konumunu normalleştirir)
    if 'PTF_Roll_Mean_168' in df.columns:
        df['Relative_Price_Pos'] = (df['PTF_Lag_24'] - df['PTF_Roll_Mean_168']) / (df['PTF_Roll_Mean_168'] + 1)

    # B. Price Momentum (Haftalık Değişim Hızı)
    df['Price_Momentum'] = df['PTF_Lag_24'] - df['PTF_Lag_168']

    # C. Net Load (Termik Santrallere Kalan Yük)
    # Toplam Yenilenebilir Enerji (Shift edilmiş verilerden!)
    ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
    existing_ren = [c for c in ren_cols if c in df.columns]
    df['Total_Renewable_Lag24'] = df[existing_ren].sum(axis=1)

    load_col = 'Yük Tahmin Planı (MWh)'
    if load_col in df.columns:
        df['Net_Load'] = df[load_col] - df['Total_Renewable_Lag24']
    else:
        df['Net_Load'] = -df['Total_Renewable_Lag24']  # Yük yoksa negatif üretim

    # D. Thermal Stress (Termik Santrallerin Yükü)
    therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
    existing_therm = [c for c in therm_cols if c in df.columns]
    df['Total_Thermal_Lag24'] = df[existing_therm].sum(axis=1)

    if load_col in df.columns:
        # (Termik Üretim / Toplam Yük) oranı
        df['Thermal_Stress'] = df['Total_Thermal_Lag24'] / (df[load_col] + 1)
    else:
        df['Thermal_Stress'] = 0

    # -------------------------------------------------------------------------
    # 5. TEMİZLİK VE FİNAL
    # -------------------------------------------------------------------------
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    print(f"🧹 Temizlik: İlk {rows_before - rows_after} satır (Lag'lerden dolayı boş) silindi.")
    print(f"✅ Modele Hazır Satır Sayısı: {rows_after}")

    return df

# =============================================================================
# KULLANIM
# =============================================================================
df_final = run_feature_engineering(df_final)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#    ADIM 4              --MODELLEME--
# =============================================================================

def run_model_training(dataframe, target_col='PTF (TL/MWH)'):
    """
    XGBoost Model Eğitimi, Tarih Bazlı Bölümleme, Optimizasyon ve Final Eğitim.
    (Orijinal kod yapısı %100 korunmuştur)
    """
    print(f"\n" + "=" * 50)
    print("🤖 ADIM 6: MODELLEME (XGBOOST) BAŞLIYOR")
    print("=" * 50)

    df = dataframe.copy()

    # -------------------------------------------------------------------------
    # 1. X (ÖZELLİKLER) ve y (HEDEF) AYRIMI
    # -------------------------------------------------------------------------
    # Modelin görmemesi gereken (Drop Listesi) sütunlar
    drop_cols = [
        'Tarih',  # Datetime formatı, model işlemez
        'Zaman',  # Datetime formatı, model işlemez
        'Saat',  # String/Object formatı veya gereksiz tekrar
        'Saat_Int',  # Hour_Sin/Cos varken bazen gereksiz olabilir
        target_col  # HEDEF DEĞİŞKEN (Sızıntıyı önlemek için X'ten atıyoruz)
    ]

    # Sadece veri setinde mevcut olanları drop listesine ekle
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    # X Matrisi (Girdiler)
    X = df.drop(columns=existing_drop_cols)

    # y Vektörü (Çıktı / Hedef)
    y = df[target_col]

    # Tarihleri Görselleştirme İçin Sakla
    if 'Tarih' in df.columns:
        dates = df['Tarih']
    else:
        dates = df.index  # Eğer tarih index'te ise

    print(f"🚫 Drop Edilen Sütunlar: {existing_drop_cols}")
    print(f"✅ X Matrisi Boyutu: {X.shape}")
    print(f"🎯 y Matrisi Boyutu: {y.shape}")

    # -------------------------------------------------------------------------
    # 2. ZAMAN SERİSİ BÖLÜMLEME (TRAIN / TEST SPLIT) - TARİH BAZLI
    # -------------------------------------------------------------------------
    train_end_date = '2025-10-31'
    test_start_date = '2025-11-01'
    test_end_date = '2025-11-30'

    # Maskeleme (Filtreleme)
    train_mask = (dates >= '2024-01-01') & (dates <= train_end_date)
    test_mask = (dates >= test_start_date) & (dates <= test_end_date)

    # Veriyi Bölme
    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # Tarihleri de ayır
    dates_train = dates.loc[train_mask]
    dates_test = dates.loc[test_mask]

    # KONTROL
    print("-" * 50)
    print(f"📉 Eğitim Seti (Train): {len(X_train)} satır")
    if len(dates_train) > 0:
        print(f"   Aralık: {dates_train.min().date()}  --->  {dates_train.max().date()}")
    print("-" * 50)
    print(f"📈 Test Seti (Test):    {len(X_test)} satır")
    if len(dates_test) > 0:
        print(f"   Aralık: {dates_test.min().date()}  --->  {dates_test.max().date()}")
    print("-" * 50)

    # Güvenlik Kontrolü
    if len(X_test) == 0:
        raise ValueError("⚠️ HATA: Test seti boş geldi! Tarih formatlarını veya veri aralığını kontrol et.")

    # -------------------------------------------------------------------------
    # 3. REFERANS NOKTASI (BENCHMARK - NAIVE FORECAST)
    # -------------------------------------------------------------------------
    if 'PTF_Lag_24' in X_test.columns:
        naive_pred = X_test['PTF_Lag_24']
        naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
        naive_mae = mean_absolute_error(y_test, naive_pred)

        print(f"🛑 Benchmark (Naive) RMSE: {naive_rmse:.2f} TL")
        print(f"🛑 Benchmark (Naive) MAE:  {naive_mae:.2f} TL")
        print("   -> Hedefimiz bu hataların altına düşmek!")
    else:
        print("⚠️ PTF_Lag_24 bulunamadı, Benchmark atlanıyor.")

    # -------------------------------------------------------------------------
    # 4. HİPERPARAMETRE OPTİMİZASYONU
    # -------------------------------------------------------------------------
    print("\n⚙️ Hiperparametre Optimizasyonu: Overfitting Önleyici Ayarlar...")

    param_dist = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.03, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0, 0.5, 1],
        'reg_lambda': [1, 5, 10],
        'objective': ['reg:squarederror']
    }

    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=1)
    tscv = TimeSeriesSplit(n_splits=10)

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=30,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)
    print(f"\n🏆 En İyi Parametreler: {random_search.best_params_}")

    # -------------------------------------------------------------------------
    # 5. FİNAL MODELİN EĞİTİLMESİ
    # -------------------------------------------------------------------------
    print("\n🦾 Final Model Eğitiliyor (Akıllı Durdurma Aktif)...")

    best_model = random_search.best_estimator_

    # Parametreyi modele ekliyoruz (set_params yöntemiyle)
    best_model.set_params(early_stopping_rounds=50)

    eval_set = [(X_train, y_train), (X_test, y_test)]

    best_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    print("✅ Model eğitimi tamamlandı.")

    # Değerleri döndür (Sonraki adımlar için gerekli)
    return best_model, X_train, X_test, y_train, y_test, dates

# =============================================================================
# KULLANIM
# =============================================================================
# Fonksiyonu çalıştır ve çıktıları değişkenlere ata
best_model, X_train, X_test, y_train, y_test, all_dates = run_model_training(df_final)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#    ADIM  5           --TAHMİN VE PERFORMANS ÖLÇÜMÜ (METRICS)--
# =============================================================================

def run_performance_evaluation(model, X_test, y_test, dates_test, naive_rmse):
    """
    Eğitilen modelin performansını ölçer, metrikleri hesaplar ve
    görselleştirme (Tahmin vs Gerçek, Feature Importance) yapar.

    Parametreler:
    model: Eğitilmiş XGBoost modeli (best_model)
    X_test: Test verisi özellikleri
    y_test: Test verisi gerçek değerleri
    dates_test: Test verisine ait tarihler
    naive_rmse: Kıyaslama yapılacak Benchmark hatası
    """
    print(f"\n" + "=" * 50)
    print("📊 ADIM 7: PERFORMANS DEĞERLENDİRME VE GRAFİKLER")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. TAHMİN YAPMA
    # -------------------------------------------------------------------------
    y_pred = model.predict(X_test)

    # Negatif tahminleri engelle (Fiyat eksi olamaz - istisnalar hariç)
    # (Senin kodundaki mantık aynen korundu)
    y_pred = np.maximum(y_pred, 0)

    # -------------------------------------------------------------------------
    # 2. METRİK HESAPLAMA (RMSE, MAE, MAPE)
    # -------------------------------------------------------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # MAPE Hesaplama (Sıfıra bölme hatasını engellemek için maskeleme yöntemi)
    # (Senin kodundaki mantık aynen korundu)
    mask = y_test != 0
    mape = (np.abs((y_test - y_pred) / y_test)[mask]).mean() * 100

    # Sonuçları Yazdır
    print("\n" + "=" * 30)
    print("📊 FİNAL MODEL SONUÇLARI")
    print("=" * 30)
    print(f"✅ Model RMSE: {rmse:.2f} TL (Hedef: < {naive_rmse:.2f})")
    print(f"✅ Model MAE:  {mae:.2f} TL")
    print(f"✅ Model MAPE: %{mape:.2f}")

    # İyileşme Oranı Hesabı
    improvement = ((naive_rmse - rmse) / naive_rmse) * 100
    print(f"🚀 Naive Modele Göre İyileşme: %{improvement:.2f}")

    # -------------------------------------------------------------------------
    # 3. GÖRSELLEŞTİRME 1: TAHMİN vs GERÇEK (ZAMAN SERİSİ)
    # -------------------------------------------------------------------------
    # Tahminleri DataFrame yapalım (Tarih indeksiyle)
    df_pred = pd.DataFrame({'Gerçek': y_test, 'Tahmin': y_pred}, index=dates_test)

    # Son 1 Haftayı (168 saat) Yakından Görelim
    last_week = df_pred.iloc[-168:]

    plt.figure(figsize=(15, 6))
    plt.plot(last_week.index, last_week['Gerçek'], label='Gerçek Fiyat (PTF)', color='blue', linewidth=2)
    plt.plot(last_week.index, last_week['Tahmin'], label='XGBoost Tahmini', color='red', linestyle='--', linewidth=2)
    plt.title('Son 1 Hafta: Gerçek vs Tahmin (Zoom In)', fontsize=14)
    plt.ylabel('PTF (TL/MWH)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # -------------------------------------------------------------------------
    # 4. GÖRSELLEŞTİRME 2: FEATURE IMPORTANCE (ÖZELLİK ÖNEMİ)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 8))

    # En önemli 20 özelliği çizdir
    # (Modelin feature_importances_ özelliğini kullanarak)
    sorted_idx = model.feature_importances_.argsort()[-20:]

    plt.barh(X_test.columns[sorted_idx], model.feature_importances_[sorted_idx], color='purple')
    plt.title("XGBoost: En Önemli Değişkenler (Feature Importance)")
    plt.xlabel("Önem Düzeyi")
    plt.show()


# =============================================================================
# KULLANIM
# =============================================================================
# Bu fonksiyonu çalıştırmak için bir önceki adımdan (run_model_training) gelen
# değişkenleri kullanacağız.

# naive_rmse değerini loglardan okuyup buraya elle yazabilirsin veya hesaplatabilirsin.
# Önceki çıktında naive_rmse 636.43 çıkmıştı.
# Ancak dinamik olması için kod içinde hesaplamak en doğrusudur.
naive_rmse_val = np.sqrt(mean_squared_error(y_test, X_test['PTF_Lag_24']))

# Fonksiyonu Çağır:
# Not: dates_test değişkenini all_dates üzerinden filtreleyerek oluşturuyoruz.
run_performance_evaluation(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    dates_test=all_dates.loc[y_test.index],
    naive_rmse=naive_rmse_val
)
# Naive (Benchmark) hatasını dinamik olarak hesapla
naive_rmse_val = np.sqrt(mean_squared_error(y_test, X_test['PTF_Lag_24']))

# Performans fonksiyonunu çalıştır
run_performance_evaluation(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    dates_test=all_dates.loc[y_test.index],
    naive_rmse=naive_rmse_val
)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#   ADIM 6       --OVERFITTING KONTOL TESTİ (TRAIN - TEST)--
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def run_overfitting_check(model, X_train, X_test, y_train, y_test):
    """
    Modelin Eğitim (Train) ve Test (Sınav) verileri arasındaki performans farkını ölçer.
    Aşırı öğrenme (Overfitting) olup olmadığını raporlar ve grafikler çizer.
    """
    print(f"\n" + "=" * 50)
    print("🔍 ADIM 8: OVERFITTING (AŞIRI ÖĞRENME) KONTROLÜ")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. SKORLARIN HESAPLANMASI
    # -------------------------------------------------------------------------
    # Eğitim Seti Tahmini
    y_train_pred = model.predict(X_train)
    y_train_pred = np.maximum(y_train_pred, 0)  # Negatif engeli

    # Test Seti Tahmini
    y_test_pred = model.predict(X_test)
    y_test_pred = np.maximum(y_test_pred, 0)

    # Hataları Hesapla (RMSE)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"📘 Eğitim (Train) Hatası (RMSE): {rmse_train:.2f} TL")
    print(f"📙 Test (Sınav) Hatası (RMSE):   {rmse_test:.2f} TL")

    # Fark Analizi
    diff = rmse_test - rmse_train
    # Sıfıra bölme hatası önlemi
    if rmse_train > 0:
        percentage_diff = (diff / rmse_train) * 100
    else:
        percentage_diff = 0

    print(f"\n⚠️ Fark: {diff:.2f} TL (%{percentage_diff:.2f})")

    # Karar Mekanizması
    if percentage_diff > 50:
        print("Sonuç: 🚨 OVERFITTING VAR! (Model eğitim setini ezberlemiş, testte zorlanıyor.)")
        print("       Öneri: 'max_depth' azaltılmalı veya 'reg_lambda' artırılmalı.")
    elif percentage_diff < 0:
        print("Sonuç: ❓ UNDERFITTING İHTİMALİ (Test sonucu eğitimden daha iyi, nadir bir durum.)")
    else:
        print("Sonuç: ✅ MODEL SAĞLIKLI (Genelleştirme yeteneği başarılı.)")

    # -------------------------------------------------------------------------
    # 2. ÖĞRENME EĞRİSİ (LEARNING CURVE) İÇİN TEKRAR EĞİTİM
    # -------------------------------------------------------------------------
    # Not: XGBoost'un eğitim geçmişini (history) alabilmek için eval_set ile
    # modelin üzerinden bir kez daha geçiyoruz (Re-fit).
    print("\n🩺 Modelin EKG'si (Öğrenme Eğrisi) Çıkarılıyor...")

    eval_set = [(X_train, y_train), (X_test, y_test)]

    # Mevcut parametreleri koruyarak tekrar fit ediyoruz ki logları alabilelim
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    results = model.evals_result()

    # Hata yoksa grafik çiz
    if results:
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        # -------------------------------------------------------------------------
        # 3. GÖRSELLEŞTİRME
        # -------------------------------------------------------------------------
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))

        # GRAFİK 1: ÖĞRENME EĞRİSİ
        ax[0].plot(x_axis, results['validation_0']['rmse'], label='Train (Eğitim)', color='blue', linewidth=2)
        ax[0].plot(x_axis, results['validation_1']['rmse'], label='Test (Sınav)', color='orange', linewidth=2,
                   linestyle='--')
        ax[0].legend()
        ax[0].set_ylabel('RMSE (Hata)')
        ax[0].set_xlabel('Ağaç Sayısı (Iterasyon)')
        ax[0].set_title('Overfitting Kontrolü: Hata Eğrileri\n(Çizgiler Birbirine Yakın ve Paralel Olmalı)')
        ax[0].grid(True, alpha=0.3)

        # GRAFİK 2: SCATTER PLOT (EZBER KONTROLÜ)
        # Noktalar ne kadar çizgi üzerindeyse o kadar iyi
        ax[1].scatter(y_train, y_train_pred, alpha=0.1, color='blue', label='Train Verisi')
        ax[1].scatter(y_test, y_test_pred, alpha=0.4, color='orange', label='Test Verisi')

        # İdeal Çizgi (45 Derece)
        lims = [0, max(y_test.max(), y_train.max())]
        ax[1].plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Tam İsabet Çizgisi')

        ax[1].set_xlabel('Gerçek Fiyat')
        ax[1].set_ylabel('Tahmin Edilen Fiyat')
        ax[1].set_title('Tahmin Tutarlılığı: Train vs Test')
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Uyarı: Model geçmiş verisi (evals_result) alınamadı, grafik çizilemiyor.")

# =============================================================================
# KULLANIM
# =============================================================================
# Bu fonksiyon, bir önceki (run_model_training) adımından gelen çıktıları kullanır.
run_overfitting_check(best_model, X_train, X_test, y_train, y_test)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#    ADIM 7  --SHAP ANALİZİ (MODEL NEDEN BU KARARI VERDİ?)--
# =============================================================================

def run_shap_analysis(model, X_test, dates_test, sample_idx=0, dependence_feature='Doğalgaz_Lag24'):
    """
    SHAP (SHapley Additive exPlanations) kullanarak modelin kararlarını açıklar.

    Parametreler:
    model: Eğitilmiş XGBoost modeli
    X_test: Test verisi (DataFrame)
    dates_test: Test verisinin tarihleri
    sample_idx: Tekil analiz (Waterfall) yapılacak satır indeksi (veya 'max' / 'random')
    dependence_feature: İlişki grafiği çizilecek özellik ismi
    """
    print(f"\n" + "=" * 50)
    print("🕵️‍♂️ ADIM 9: SHAP ANALİZİ (KARAR MEKANİZMASI)")
    print("=" * 50)

    # 1. Explainer Oluşturma
    # TreeExplainer, ağaç tabanlı modeller için en hızlısıdır.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"❌ SHAP hesaplanırken hata oluştu: {e}")
        return

    # -------------------------------------------------------------------------
    # GRAFİK 1: SHAP SUMMARY PLOT (GENEL BAKIŞ)
    # -------------------------------------------------------------------------
    print("\n1. Özet Grafik (Summary Plot) Çiziliyor...")
    print("   (Kırmızı: Yüksek Değer, Mavi: Düşük Değer -> Sağa: Fiyat Artışı, Sola: Düşüş)")

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Özeti: Hangi Özellik Fiyatı Nasıl Etkiliyor?", fontsize=16)
    plt.show()

    # -------------------------------------------------------------------------
    # GRAFİK 2: WATERFALL PLOT (TEKİL ANALİZ)
    # -------------------------------------------------------------------------
    # İndeks Belirleme Mantığı
    idx = 0
    if sample_idx == 'max':
        # Modelin en yüksek fiyat tahmin ettiği saati bul
        preds = model.predict(X_test)
        idx = np.argmax(preds)
        print(f"\n📍 Analiz Modu: EN YÜKSEK FİYAT TAHMİNİ SEÇİLDİ (İndeks: {idx})")
    elif sample_idx == 'random':
        idx = np.random.randint(0, len(X_test))
        print(f"\n📍 Analiz Modu: RASTGELE SAAT SEÇİLDİ (İndeks: {idx})")
    else:
        idx = int(sample_idx)

    # Tarih ve Tahmin Bilgisi
    # predict fonksiyonu numpy array dönebilir, tek değeri almak için [idx]
    current_pred = model.predict(X_test.iloc[[idx]])[0]
    current_date = dates_test.iloc[idx]

    print(f"\n2. Tekil Tahmin Analizi (Waterfall Plot)")
    print(f"   🔍 İncelenen Tarih: {current_date}")
    print(f"   🔍 Modelin Tahmini: {current_pred:.2f} TL")

    plt.figure(figsize=(10, 6))
    # max_display=15: En etkili 15 nedeni göster
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f"Fiyat Neden Böyle Çıktı? ({current_date})", fontsize=14)
    plt.show()

    # -------------------------------------------------------------------------
    # GRAFİK 3: DEPENDENCE PLOT (İLİŞKİ ANALİZİ)
    # -------------------------------------------------------------------------
    print(f"\n3. Bağımlılık Grafiği (Dependence Plot): {dependence_feature}")

    if dependence_feature in X_test.columns:
        plt.figure(figsize=(10, 6))
        # interaction_index='auto': SHAP, renklendirmek için en alakalı ikinci değişkeni otomatik seçer
        shap.plots.scatter(shap_values[:, dependence_feature], color=shap_values, show=False)
        plt.title(f"İlişki Analizi: {dependence_feature} vs Fiyat Etkisi", fontsize=14)
        plt.show()
    else:
        print(f"⚠️ Uyarı: '{dependence_feature}' sütunu bulunamadı, grafik atlanıyor.")

    print("\n✅ SHAP Analizi Tamamlandı.")

# =============================================================================
# KULLANIM (DÜZELTİLMİŞ)
# =============================================================================

# 1. Test Tarihlerini Güvenli Şekilde Hazırla (İndeks hatasını önlemek için)
# X_test'i oluştururken kullandığımız tarih aralığının aynısını kullanıyoruz.
dates_test = all_dates[(all_dates >= '2025-11-01') & (all_dates <= '2025-11-30')]

# 2. Fonksiyonu Çalıştır
run_shap_analysis(
    model=best_model,
    X_test=X_test,
    dates_test=dates_test,  # Düzeltilmiş tarih serisi
    sample_idx='max',       # En yüksek fiyatlı saati inceler
    dependence_feature='Doğalgaz_Lag24'
)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#  ADIM 8 --ARALIK 2025 SENARYO TAHMİNİ (FİNAL DÜZELTİLMİŞ VE BİRLEŞTİRİLMİŞ SÜRÜM)--
# =============================================================================

def run_forecast_december(model, X_last_month, y_last_month, dates_last_month):
    """
    Eğitilen modeli kullanarak Aralık 2025 için saatlik tahminler üretir.
    Özyinelemeli (Recursive) tahmin mantığı kullanılır.

    Parametreler:
    model: Eğitilmiş XGBoost modeli (best_model)
    X_last_month: Son ayın (Kasım) özellik verisi (X_test)
    y_last_month: Son ayın gerçek fiyatları (y_test)
    dates_last_month: Son ayın tarihleri
    """
    print(f"\n" + "=" * 50)
    print("🔮 ADIM 10: ARALIK 2025 SENARYO TAHMİNİ")
    print("=" * 50)

    # 1. ARALIK AYI İÇİN BOŞ ŞABLON OLUŞTURMA
    # -------------------------------------------------------------------------
    future_dates = pd.date_range(start='2025-12-01 00:00', end='2025-12-31 23:00', freq='h')
    print(f"📅 Hedef Dönem: {len(future_dates)} Saat ({future_dates.min()} - {future_dates.max()})")

    # X_test verisinden kopya al (Şablon olarak kullanacağız)
    temp_X = X_last_month.copy()

    # Satır sayısını eşitleme (720 -> 744 saat)
    missing_hours = len(future_dates) - len(temp_X)

    if missing_hours > 0:
        # Eksik kısım kadar veriyi son günden kopyala ekle
        padding = temp_X.iloc[-missing_hours:].copy()
        future_X = pd.concat([temp_X, padding], axis=0)
    else:
        # Fazlaysa kes (Nadir durum)
        future_X = temp_X.iloc[-len(future_dates):].copy()

    # İndeksi Aralık ayı yap
    future_X.index = future_dates

    # 2. TARİHSEL ÖZELLİKLERİ GÜNCELLEME
    # -------------------------------------------------------------------------
    print("⚙️ Tarih ve Tatil özellikleri güncelleniyor...")

    # Geçici 'Saat_Int' oluştur (Sin/Cos hesabı için)
    future_X['Saat_Int'] = future_dates.hour

    # Takvim özellikleri
    if 'Month' in future_X.columns: future_X['Month'] = 12
    future_X['Day_of_Week'] = future_dates.dayofweek
    future_X['Is_Weekend'] = future_X['Day_of_Week'].isin([5, 6]).astype(int)

    # Trigonometrik Dönüşümler
    if 'Hour_Sin' in future_X.columns:
        future_X['Hour_Sin'] = np.sin(2 * np.pi * future_X['Saat_Int'] / 24)
        future_X['Hour_Cos'] = np.cos(2 * np.pi * future_X['Saat_Int'] / 24)
    if 'Day_Sin' in future_X.columns:
        future_X['Day_Sin'] = np.sin(2 * np.pi * future_X['Day_of_Week'] / 7)
        future_X['Day_Cos'] = np.cos(2 * np.pi * future_X['Day_of_Week'] / 7)

    # Tatil Günleri
    tr_holidays = holidays.TR(years=[2025])
    if 'Is_Holiday' in future_X.columns:
        future_X['Is_Holiday'] = future_dates.to_series().apply(lambda x: 1 if x in tr_holidays else 0)

    # Temizlik (Model eğitilirken olmayan sütunları at)
    if 'Saat_Int' in future_X.columns:
        future_X.drop(columns=['Saat_Int'], inplace=True)

    # 3. ÖZYİNELEMELİ TAHMİN DÖNGÜSÜ (RECURSIVE FORECASTING)
    # -------------------------------------------------------------------------
    print("⏳ Simülasyon Başlıyor (Bu işlem biraz sürebilir)...")

    future_preds = []
    # Başlangıç hafızası: Kasım ayının son 1 haftası
    last_known_prices = y_last_month.iloc[-168:].values.tolist()

    for i in range(len(future_X)):
        # Tek satır al (DataFrame olarak kalmalı)
        current_row = future_X.iloc[[i]].copy()

        # --- DİNAMİK GÜNCELLEME (Feature Engineering'in Devamı) ---
        # Model tahmini yapabilmek için "Dün fiyat neydi?" sorusunun cevabını
        # bir önceki tahminimizden alıp buraya koymalıyız.

        # Lag 24 (Dün)
        if 'PTF_Lag_24' in current_row.columns:
            current_row['PTF_Lag_24'] = last_known_prices[-24]

        # Lag 168 (Geçen Hafta)
        if 'PTF_Lag_168' in current_row.columns:
            current_row['PTF_Lag_168'] = last_known_prices[-168]

        # Hareketli Ortalamalar
        if 'PTF_Roll_Mean_24' in current_row.columns:
            current_row['PTF_Roll_Mean_24'] = np.mean(last_known_prices[-24:])

        # Sniper Özellikler (Rasyolar)
        if 'Relative_Price_Pos' in current_row.columns:
            roll_168 = np.mean(last_known_prices[-168:])
            denom = roll_168 if roll_168 != 0 else 1
            current_row['Relative_Price_Pos'] = (current_row['PTF_Lag_24'] - roll_168) / denom

        if 'Price_Momentum' in current_row.columns:
            current_row['Price_Momentum'] = current_row['PTF_Lag_24'] - current_row['PTF_Lag_168']

        # TAHMİN YAP
        pred = model.predict(current_row)[0]
        pred = max(0, pred)  # Negatif fiyat engeli

        # Tahmini listeye ekle (Gelecek adımlar için hafızaya al)
        future_preds.append(pred)
        last_known_prices.append(pred)

    print("✅ Aralık ayı tahmini tamamlandı.")

    # 4. SONUÇLARI KAYDETME VE GÖRSELLEŞTİRME
    # -------------------------------------------------------------------------
    df_forecast = pd.DataFrame({'Tahmin_Aralik': future_preds}, index=future_dates)

    # İstatistiksel Özet
    print(f"\n📢 Aralık 2025 Tahmin Özeti:")
    print(f"   Min Fiyat: {df_forecast['Tahmin_Aralik'].min():.2f} TL")
    print(f"   Max Fiyat: {df_forecast['Tahmin_Aralik'].max():.2f} TL")
    print(f"   Ort Fiyat: {df_forecast['Tahmin_Aralik'].mean():.2f} TL")

    # Grafik Çizimi
    plt.figure(figsize=(16, 6))

    # Geçmiş (Kasım Sonu - Mavi)
    last_week_dates = dates_last_month.iloc[-168:]
    last_week_values = y_last_month.iloc[-168:]

    plt.plot(last_week_dates, last_week_values, label='Gerçekleşen (Kasım Sonu)', color='navy', alpha=0.7)

    # Gelecek (Aralık - Kırmızı)
    plt.plot(df_forecast.index, df_forecast['Tahmin_Aralik'], label='Forecast (Aralık 2025)', color='red')

    # Ortalama Çizgisi
    plt.axhline(df_forecast['Tahmin_Aralik'].mean(), color='green', linestyle='--', label='Aralık Ortalaması')

    plt.title('Aralık 2025: Gelecek Fiyat Tahmin Senaryosu')
    plt.ylabel('PTF (TL/MWH)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Tarih formatı
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.gcf().autofmt_xdate()
    plt.show()

    # 5. EXCEL KAYDI (OPSİYONEL AMA ÖNEMLİ)
    # -------------------------------------------------------------------------
    try:
        df_forecast.to_excel("Aralik_2025_Tahminleri.xlsx")
        print("\n💾 Dosya Kaydedildi: Aralik_2025_Tahminleri.xlsx")
    except:
        print("\n⚠️ Uyarı: Excel dosyası kaydedilemedi (Dosya açık olabilir).")

    return df_forecast

# =============================================================================
# KULLANIM
# =============================================================================
# Bu fonksiyonu çalıştırmak için 1. tarihler, 2. X_test ve 3. y_test gereklidir.
# dates_test değişkenini daha önce oluşturmuştuk.

dates_test_fixed = all_dates[(all_dates >= '2025-11-01') & (all_dates <= '2025-11-30')]

df_aralik_tahmin = run_forecast_december(
    model=best_model,
    X_last_month=X_test,
    y_last_month=y_test,
    dates_last_month=dates_test_fixed
)
#////////////////////////////////////////////////////////////////////////////////////









# =============================================================================
#   ADIM 9    --RESİDUAL ANALİZİ VE GÜVENİLİRLİK TESTİ--
# =============================================================================
# =============================================================================
# ADIM 9.1: RESIDUAL (HATA) ANALİZİ
# =============================================================================
def run_residual_analysis(y_test, y_pred):
    """
    Modelin hata analizini yapar, istatistiksel metrikleri hesaplar,
    4'lü tanı grafiği çizer ve modelin güvenilirliğini yorumlar.

    Geri Döndürür: residuals (Hata serisi)
    """
    print(f"\n" + "=" * 50)
    print("🕵️‍♂️ ADIM 9: MODEL HATA ANALİZİ VE GÜVENİLİRLİK TESTİ")
    print("=" * 50)

    # Not: Geleceğin gerçeğini bilmediğimiz için 'Test Seti' üzerinden analiz yapıyoruz.

    # 1. Hataları Hesapla
    # -----------------------------------------------------------------------------
    residuals = y_test - y_pred

    # İstatistiksel Metrikler (İki kodun birleşimi)
    mean_resid = np.mean(residuals)
    std_resid = np.std(residuals)
    min_resid = np.min(residuals)
    max_resid = np.max(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    dw_score = durbin_watson(residuals)

    print(f"📊 İSTATİSTİKSEL ÖZET:")
    print(f"   Hata Ortalaması (Bias): {mean_resid:.2f} TL (0'a ne kadar yakınsa o kadar iyi)")
    print(f"   Standart Sapma:         {std_resid:.2f}")
    print(f"   Min Hata / Max Hata:    {min_resid:.2f} / {max_resid:.2f}")
    print(f"   Çarpıklık (Skewness):   {skewness:.2f} (0 ideal)")
    print(f"   Basıklık (Kurtosis):    {kurtosis:.2f} (Yüksekse 'Şişman Kuyruk' var demektir)")
    print(f"   Durbin-Watson Score:    {dw_score:.2f} (2.00 İdeal, 1.5-2.5 arası kabul)")

    # 2. GÖRSELLEŞTİRME (4'lü Panel)
    # -----------------------------------------------------------------------------
    # Seaborn stilini ayarla (Daha şık görünüm için)
    sns.set(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Model Güvenilirlik Testi (Residual Diagnostics)', fontsize=16, fontweight='bold')

    # GRAFİK A: Residuals vs Time (Hataların Zamana Göre Dağılımı)
    axes[0, 0].plot(residuals.index, residuals, color='purple', alpha=0.7, linewidth=1)
    axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('1. Hataların Zaman İçindeki Değişimi (Rastgele Olmalı)')
    axes[0, 0].set_ylabel('Hata (TL)')

    # GRAFİK B: Residuals vs Predicted (Heteroskedasite Kontrolü)
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, color='teal', edgecolor='k', s=30)
    axes[0, 1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_title('2. Hata vs Tahmin (Heteroskedasite Kontrolü)')
    axes[0, 1].set_xlabel('Tahmin Edilen Fiyat')
    axes[0, 1].set_ylabel('Hata')

    # GRAFİK C: Histogram (Hata Dağılımı)
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='orange', bins=40, line_kws={'linewidth': 2})
    axes[1, 0].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[1, 0].set_title('3. Hata Dağılımı (Çan Eğrisi Beklenir)')
    axes[1, 0].set_xlabel('Hata Miktarı (TL)')

    # GRAFİK D: Q-Q Plot (Normallik Testi)
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set_color('blue')  # Noktalar
    axes[1, 1].get_lines()[0].set_markersize(5)
    axes[1, 1].get_lines()[1].set_color('red')  # İdeal Çizgi
    axes[1, 1].get_lines()[1].set_linewidth(2)
    axes[1, 1].set_title('4. Q-Q Plot (Normallik Testi)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. YORUM VE SONUÇ RAPORU (Otomatik Yorumlama)
    # -----------------------------------------------------------------------------
    print("\n📢 MODEL GÜVENİLİRLİK RAPORU:")

    # Bias (Yanlılık) Kontrolü
    if abs(mean_resid) < 50:
        print("   ✅ BAŞARILI (Bias): Modelin hata ortalaması 0'a yakın. Yanlılık yok.")
    else:
        print("   ⚠️ UYARI (Bias): Modelde sistematik bir kayma var.")

    # Skewness (Simetri) Kontrolü
    if abs(skewness) < 0.5:
        print("   ✅ BAŞARILI (Simetri): Hatalar Normal dağılıma yakın. Model tarafsız.")
    elif skewness > 0:
        print("   ⚠️ UYARI (Simetri): Pozitif Çarpıklık. Model fiyatları bazen olduğundan DÜŞÜK tahmin ediyor.")
    else:
        print("   ⚠️ UYARI (Simetri): Negatif Çarpıklık. Model fiyatları bazen olduğundan YÜKSEK tahmin ediyor.")

    # Kurtosis (Uç Değer) Kontrolü
    if kurtosis > 3:
        print("   ℹ️ BİLGİ (Uç Değerler): 'Şişman Kuyruk' var. Model nadiren de olsa büyük hata (Spike) yapabilir.")

        # --- YENİ YORUM: DURBIN-WATSON ---
        if 1.5 <= dw_score <= 2.5:
            print("   ✅ BAŞARILI (Otokorelasyon): Durbin-Watson skoru ideal. Hatalar bağımsız.")
        else:
            print(f"   ⚠️ UYARI (Otokorelasyon): Durbin-Watson {dw_score:.2f}. Hatalar arasında ilişki olabilir.")

    # Hesaplanan hataları geri döndür (Belki Excel'e kaydetmek istersin)
    return residuals


# =============================================================================
# ADIM 9.2:   MODEL GÜVENİLİRLİK VE ROBUSTNESS (SAĞLAMLIK) TESTİ
# =============================================================================

# Uyarıları kapatalım (Temiz çıktı için)
warnings.filterwarnings("ignore")

def run_reliability_tests(model, X_full, y_full, date_series):
    """
    Modeli zorlu şartlarda test eder:
    1. Mevsimsel Backtest (Farklı aylarda nasıl?)
    2. Duyarlılık (Sensitivity) (Girdiler değişince tepki veriyor mu?)
    3. Stres Testi (Ekstrem senaryolar)
    4. Güven Aralığı Grafiği
    """
    print("\n" + "=" * 50)
    print("🛡️ ADIM 9.5: MODEL GÜVENİLİRLİK VE ROBUSTNESS RAPORU")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # TEST 1: BACKTESTING (MEVSİMSEL DAYANIKLILIK TESTİ)
    # -------------------------------------------------------------------------
    print("\n🧪 TEST 1: BACKTESTING (Mevsimsel Kontrol)")
    print("-" * 40)

    # Test edilecek dönemler (Veri setinde bu tarihlerin olduğundan emin olmalıyız)
    test_periods = [
        ("🌸 İlkbahar (Nisan 2025)", '2025-04-01', '2025-04-30'),
        ("☀️ Yaz Zirvesi (Temmuz 2025)", '2025-07-01', '2025-07-31'),
        ("🍂 Sonbahar/Test (Kasım 2025)", '2025-11-01', '2025-11-30')
    ]

    for label, start_date, end_date in test_periods:
        # Tarih maskesi oluştur
        mask = (date_series >= start_date) & (date_series <= end_date)

        if mask.sum() == 0:
            print(f"⚠️ {label}: Veri bulunamadı! (Atlanıyor)")
            continue

        X_period = X_full.loc[mask]
        y_period = y_full.loc[mask]

        # Tahmin
        preds = model.predict(X_period)
        preds = np.maximum(preds, 0)

        # Metrikler
        if len(y_period) > 0:
            rmse_period = np.sqrt(mean_squared_error(y_period, preds))
            # +1 sıfıra bölme hatası için
            mape_period = np.mean(np.abs((y_period - preds) / (y_period + 1))) * 100
            print(f"📅 {label:<30} | RMSE: {rmse_period:.2f} TL | MAPE: %{mape_period:.2f}")
        else:
            print(f"⚠️ {label}: Veri yetersiz.")

    print("\n✅ YORUM: MAPE değerleri %15-25 bandındaysa model mevsimsellikten etkilenmiyor demektir.")

    # -------------------------------------------------------------------------
    # TEST 2: SENSITIVITY ANALYSIS (DUYARLILIK ANALİZİ)
    # -------------------------------------------------------------------------
    print("\n🧪 TEST 2: SENSITIVITY (Duyarlılık Analizi)")
    print("-" * 40)

    # Test için Kasım ayını baz alalım (En güncel ve stabil)
    mask_nov = (date_series >= '2025-11-01') & (date_series <= '2025-11-30')

    if mask_nov.sum() > 0:
        X_test_sample = X_full.loc[mask_nov].copy()
        base_preds = model.predict(X_test_sample)
        base_mean = np.mean(base_preds)

        # Değiştirilecek Kritik Kolonlar
        target_cols = ['Yük Tahmin Planı (MWh)', 'Dolar_Kuru', 'Doğalgaz_Lag24']

        for col in target_cols:
            if col in X_test_sample.columns:
                # Senaryo: Değişkeni %10 artır (Ceteris Paribus)
                X_shocked = X_test_sample.copy()
                X_shocked[col] = X_shocked[col] * 1.10

                shocked_preds = model.predict(X_shocked)
                shocked_mean = np.mean(shocked_preds)

                change_pct = ((shocked_mean - base_mean) / base_mean) * 100

                # Yön kontrolü
                direction = "⬆️ Artış" if change_pct > 0 else "⬇️ Düşüş"
                # Mantık: Yük ve Dolar artarsa fiyat artmalı
                logic = "✅ Mantıklı" if change_pct > 0 else "❓ İlginç"

                print(f"📊 {col:<25} (+%10) -> Fiyat Etkisi: %{change_pct:+.2f} ({direction}) {logic}")
            else:
                print(f"⚠️ {col} sütunu bulunamadı, atlanıyor.")
    else:
        print("⚠️ Kasım ayı verisi bulunamadığı için Sensitivity testi yapılamadı.")

    # -------------------------------------------------------------------------
    # TEST 3: SCENARIO ANALYSIS (EKSTREM DURUM TESTİ)
    # -------------------------------------------------------------------------
    print("\n🧪 TEST 3: SCENARIO ANALYSIS (Stres Testi)")
    print("-" * 40)

    # Ortalama bir satır alıp sadece ilgilendiğimiz değerleri değiştireceğiz
    base_row = X_full.mean().to_frame().T

    # Senaryo 1: KIŞ GECESİ KABUSU (Yüksek Yük, Düşük Rüzgar, Pahalı Gaz)
    nightmare_row = base_row.copy()
    if 'Yük Tahmin Planı (MWh)' in base_row.columns: nightmare_row['Yük Tahmin Planı (MWh)'] = 50000
    if 'Rüzgar_Lag24' in base_row.columns: nightmare_row['Rüzgar_Lag24'] = 100
    if 'Doğalgaz_Lag24' in base_row.columns: nightmare_row['Doğalgaz_Lag24'] = 15000

    # Senaryo 2: BAHAR BAYRAMI (Düşük Yük, Yüksek Yenilenebilir)
    paradise_row = base_row.copy()
    if 'Yük Tahmin Planı (MWh)' in base_row.columns: paradise_row['Yük Tahmin Planı (MWh)'] = 20000
    if 'Rüzgar_Lag24' in base_row.columns: paradise_row['Rüzgar_Lag24'] = 8000
    if 'Güneş_Lag24' in base_row.columns: paradise_row['Güneş_Lag24'] = 5000

    try:
        pred_nightmare = model.predict(nightmare_row)[0]
        pred_paradise = model.predict(paradise_row)[0]

        print(f"🔥 Kabus Senaryosu (Yüksek Talep/Az Rüzgar): {pred_nightmare:.2f} TL")
        print(f"🌼 Cennet Senaryosu (Düşük Talep/Bol Güneş):  {pred_paradise:.2f} TL")

        if pred_nightmare > pred_paradise * 1.5:
            print("✅ SONUÇ: Model piyasa fizik kurallarını kavramış. Kıtlıkta fiyatı uçuruyor.")
        else:
            print("⚠️ SONUÇ: Model ekstrem durumlara yeterince sert tepki vermiyor.")
    except Exception as e:
        print(f"⚠️ Stres testi sırasında hata: {e}")

    # -------------------------------------------------------------------------
    # TEST 4: CONFIDENCE INTERVALS (GÜVEN ARALIĞI)
    # -------------------------------------------------------------------------
    print("\n🧪 TEST 4: GÜVEN ARALIĞI (Son 1 Hafta)")
    print("-" * 40)

    # Son 1 haftayı bul
    last_date = date_series.max()
    first_date_viz = last_date - pd.Timedelta(days=7)

    mask_viz = (date_series >= first_date_viz) & (date_series <= last_date)

    if mask_viz.sum() > 0:
        X_viz = X_full.loc[mask_viz]
        y_viz = y_full.loc[mask_viz]
        dates_viz = date_series.loc[mask_viz]

        preds_viz = model.predict(X_viz)
        preds_viz = np.maximum(preds_viz, 0)

        # Modelin genel hatasını (RMSE) baz alarak bant çiziyoruz
        # (Burada manuel 452 yerine dinamik hesaplama yapabiliriz ama orijinal koda sadık kaldım)
        rmse_viz = np.sqrt(mean_squared_error(y_viz, preds_viz))
        confidence_interval = 1.96 * rmse_viz  # %95 Güven Aralığı

        lower_bound = preds_viz - confidence_interval
        upper_bound = preds_viz + confidence_interval
        lower_bound = np.maximum(lower_bound, 0)

        plt.figure(figsize=(15, 7))
        plt.plot(dates_viz, y_viz, label='Gerçekleşen', color='black', linewidth=2)
        plt.plot(dates_viz, preds_viz, label='Tahmin', color='blue', linestyle='--')

        # Güven aralığını boya
        plt.fill_between(dates_viz, lower_bound, upper_bound, color='blue', alpha=0.2,
                         label=f'%95 Güven Aralığı (+/- {confidence_interval:.0f} TL)')

        plt.title('Model Güvenilirlik Bandı (Son 1 Hafta)', fontsize=14)
        plt.ylabel('PTF (TL/MWh)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        print("✅ Grafik çizildi. Mavi alan, modelin güvenli limanıdır.")
    else:
        print("⚠️ Görselleştirme için yeterli veri yok.")

    print("\n✅ Güvenilirlik testleri tamamlandı.")

# =============================================================================
# KULLANIM
# =============================================================================
# 1. PARÇALARI BİRLEŞTİR (X ve y)
# -----------------------------------------------------------------------------
# Eğitim ve Test setlerini alt alta ekleyerek bütün veriyi elde ediyoruz.
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# 2. TARİHLERİ EŞLEŞTİR (KRİTİK DÜZELTME 🛠️)
# -----------------------------------------------------------------------------
# Hata burada çıkıyordu. 'dates_train' falan aramak yerine,
# Elimizdeki 'y_full'un indeksini kullanarak ana tarih listesinden (all_dates)
# doğru tarihleri çekip alıyoruz. En güvenli yöntem budur.

# all_dates değişkeni Adım 6'dan (run_model_training) gelmiş olmalı.
# Eğer adı farklıysa (örn: dates) burayı ona göre değiştir.
full_dates_aligned = all_dates.loc[y_full.index]

# 3. TAHMİN ÜRET
# -----------------------------------------------------------------------------
# Residual analizi için test seti tahminlerini hazırlayalım.
y_pred_final = best_model.predict(X_test)
y_pred_final = np.maximum(y_pred_final, 0) # Negatif fiyat koruması

# 4. FONKSİYONLARI ÇALIŞTIR
# -----------------------------------------------------------------------------
print(f"\n🚀 Adım 9 Başlatılıyor...")
print(f"   Analiz edilecek toplam veri sayısı: {len(X_full)} satır")

# A) Güvenilirlik Testi (Tüm yıl için)
run_reliability_tests(best_model, X_full, y_full, full_dates_aligned)

# B) Residual (Hata) Analizi (Sadece Test ayı için)
residuals = run_residual_analysis(y_test, y_pred_final)
#////////////////////////////////////////////////////////////////////////////////////
