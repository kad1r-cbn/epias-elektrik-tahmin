import locale
import datetime
from statistics import quantiles
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import numpy as np
import pandas as pd

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

##############################################################
#---------------------------
# EDA
#---------------------------

def data_summary(dataframe, head=5):
    print("######### Shape ########")
    print(dataframe.shape)
    print("######### Type ########")
    print(dataframe.dtypes)
    print("######### Head #######")
    print(dataframe.head(head))
    print("######### Tail #######")
    print(dataframe.tail(head))
    print("######### Nan #######")
    print(dataframe.isnull().sum())

data_summary(df_final)


df_final.loc[df_final['Güneş'] < 0, 'Güneş'] = 0

print("Güneş değeri dağılımı:")
print("Negatif (<0):", (df_final['Güneş'] < 0).sum())
print("Sıfır (=0):", (df_final['Güneş'] == 0).sum())
print("Pozitif (>0):", (df_final['Güneş'] > 0).sum())
df_final['Güneş'] = df_final['Güneş'].clip(lower=0)
def degisken_analiz(dataframe, cat_th=2, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = degisken_analiz(df_final)

def numeric_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

print("\n--- NUMERİK DEĞİŞKENLERİN DAĞILIMI ---")
for col in num_cols:
    numeric_summary(df_final, numerical_col=col, plot=True)


#---------------------------
# Target
#---------------------------

def target_summary_with_numeric(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_numeric(df_final, "PTF (TL/MWH)", col)

#---------------------------
# Korelasyon
#---------------------------

df_final[num_cols].corr()

f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df_final[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block = True)

#---------------------------
# Grafik ile Analiz
#---------------------------
def check_physical_integrity(df):
    print("🕵️‍♂️ Fiziksel Tutarlılık Kontrolü Yapılıyor...")

    # 1. Negatif Üretim Kontrolü (İmkansız Olay)
    prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit']
    # Veri setinde olanları seç
    existing_cols = [c for c in prod_cols if c in df.columns]

    for col in existing_cols:
        negatives = df[df[col] < 0]
        if len(negatives) > 0:
            print(f"⚠️ UYARI: {col} sütununda {len(negatives)} adet negatif değer var! 0'a eşitleniyor.")
            df.loc[df[col] < 0, col] = 0
        else:
            print(f"✅ {col}: Temiz (Negatif yok).")

    # 2. PTF Kontrolü (Hata vs Gerçek Ayrımı)
    # Tavan Fiyatı manuel belirleyebiliriz (Örn: 2025 için 5000 TL diyelim, teyit etmen lazım)
    MAX_PRICE_LIMIT = 6000
    MIN_PRICE_LIMIT = 0

    errors = df[(df['PTF (TL/MWH)'] > MAX_PRICE_LIMIT) | (df['PTF (TL/MWH)'] < MIN_PRICE_LIMIT)]
    if len(errors) > 0:
        print(f"🚨 KRİTİK: PTF sütununda {len(errors)} adet mantıksız (Tavan üstü veya Negatif) değer var!")
        # Bunları baskılamıyoruz, SİLİYORUZ. Çünkü gerçek mi hata mı bilemeyiz.
        # df = df.drop(errors.index) # İstersen silebilirsin
    else:
        print("✅ PTF: Mantıksız uç değer (Error) görünmüyor.")

    print("-" * 30)
    return df
check_physical_integrity(df_final)


def plot_all_boxplots(df):
    # Stil ayarları
    sns.set_theme(style="whitegrid")

    # 1. GRUP: Fiyat Değişkenleri (Küçük Ölçekli)
    # PTF, Dolar ve Doğalgaz Fiyatları benzer ölçeklerdedir.
    price_cols = ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh']

    # 2. GRUP: Büyük Ölçekli Üretim ve Yük
    # Yük tahmini ve ana üretim kalemleri (Barajlı, Doğalgaz Üretimi)
    large_scale_cols = ['Yük Tahmin Planı (MWh)', 'Doğalgaz', 'Barajlı', 'İthal Kömür']

    # 3. GRUP: Yenilenebilir ve Diğer Üretimler
    # Rüzgar, Güneş, Akarsu, Jeotermal gibi daha orta ölçekli üretimler
    renewable_cols = ['Rüzgar', 'Güneş', 'Akarsu', 'Linyit', 'Jeotermal', 'Biyokütle', 'Fuel Oil']

    # Grafiklerin çizilmesi
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Plot 1: Fiyatlar
    sns.boxplot(data=df[price_cols], ax=axes[0], palette="Set2")
    axes[0].set_title('Grup 1: Fiyat Bazlı Değişkenler', fontsize=15)

    # Plot 2: Büyük Ölçekli Veriler
    sns.boxplot(data=df[large_scale_cols], ax=axes[1], palette="Set1")
    axes[1].set_title('Grup 2: Yük ve Büyük Ölçekli Üretimler', fontsize=15)

    # Plot 3: Yenilenebilir ve Diğerleri
    sns.boxplot(data=df[renewable_cols], ax=axes[2], palette="Pastel1")
    axes[2].set_title('Grup 3: Yenilenebilir Enerji ve Diğer Üretimler', fontsize=15)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

plot_all_boxplots(df_final)

df_final['PTF (TL/MWH)'].describe().T
sayi = (df_final['PTF (TL/MWH)'] == 99999.000).sum()
print(f"99999.000 değeri {sayi} kez geçiyor.")

#------------------------------------------------------------------------------------------------------------
# İSTATİKSEL TESTLER
#------------------------------------------------------------------------------------------------------------

#---------------------------
# NORMALLİK TESTİ
#---------------------------
# Veriyi normalize ederek K-S testi yapalım
ptf_clean = df_final['PTF (TL/MWH)'].dropna()
ks_stat, p_value_ks = stats.kstest((ptf_clean - ptf_clean.mean()) / ptf_clean.std(), 'norm')

print(f"K-S Testi p-değeri: {p_value_ks}")

plt.figure(figsize=(10, 6))
# Gerçek verinin dağılımı
sns.histplot(df_final['PTF (TL/MWH)'], kde=True, stat="density", color='skyblue', label='Gerçek Dağılım')

# İdeal Normal Dağılım eğrisi (Karşılaştırma için)
mu, std = df_final['PTF (TL/MWH)'].mean(), df_final['PTF (TL/MWH)'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label='Teorik Normal Dağılım')

plt.title('PTF Dağılımı vs Teorik Normal Dağılım')
plt.legend()
plt.show()

fig = sm.qqplot(df_final['PTF (TL/MWH)'].dropna(), line='s')
plt.title('PTF İçin Q-Q Plot')
plt.show()

print(f"Skewness (Çarpıklık): {df_final['PTF (TL/MWH)'].skew()}")
print(f"Kurtosis (Basıklık): {df_final['PTF (TL/MWH)'].kurt()}")


#---------------------------
# DURAĞANLIK TESTİ
#---------------------------

# ADF Testini çalıştır
# autolag='AIC' parametresi en iyi gecikme (lag) sayısını otomatik seçer
adf_test = adfuller(df_final['PTF (TL/MWH)'].dropna(), autolag='AIC')

print(f"ADF İstatistiği: {adf_test[0]}")
print(f"p-değeri: {adf_test[1]}")
print("Kritik Değerler:")
for key, value in adf_test[4].items():
    print(f"\t{key}: {value}")

if adf_test[1] <= 0.05:
    print("\nSonuç: p <= 0.05. H0 reddedilir. Seri DURAĞANDIR.")
else:
    print("\nSonuç: p > 0.05. H0 reddedilemez. Seri DURAĞAN DEĞİLDİR (Trend var).")

# Bağımsız değişken listesi (PTF hariç, sadece sayısal olanlar)
independent_cols = [col for col in df_final.columns if
                    col not in ['PTF (TL/MWH)', 'Tarih', 'Zaman'] and df_final[col].dtype in ['float64', 'int64']]

adf_results = []


for col in independent_cols:
    # NaN değerleri temizleyerek testi çalıştır
    series = df_final[col].dropna()
    result = adfuller(series, autolag='AIC')

    p_value = result[1]
    is_stationary = "Evet" if p_value <= 0.05 else "Hayır"

    adf_results.append({
        'Değişken': col,
        'ADF İstatistiği': round(result[0], 4),
        'p-değeri': p_value,
        'Durağan mı?': is_stationary
    })

# Sonuçları DataFrame olarak görselleştir
adf_df = pd.DataFrame(adf_results)
print(adf_df)


# Durağan olmayan ve sınırda olan değişkenleri görselleştirelim
cols_to_plot = ['Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh', 'Akarsu', 'Jeotermal']

fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 15))

for i, col in enumerate(cols_to_plot):
    axes[i].plot(df_final.index, df_final[col], color='tab:blue')
    axes[i].set_title(f'{col} - Zaman Serisi Grafiği (Durağanlık Kontrolü)')
    axes[i].set_ylabel('Değer')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#---------------------------
# VOLALİTE
#---------------------------

# 1. Grup: Fiyat ve Maliyet Volatilitesi (PTF, Dolar, Gaz Fiyatı)
# Not: Doların ham fiyatı trend izlese de, volatilitesi ekonomik risk dönemlerini gösterir.
price_maliye_cols = ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh']

# 2. Grup: Esnek ve Baz Yük Üretim Volatilitesi (Fosil Yakıtlar)
# Doğalgaz ve Kömür santrallerindeki ani oynaklıklar sistemdeki arz şoklarını temsil eder.
fosil_cols = ['Doğalgaz', 'Linyit', 'İthal Kömür']

# 3. Grup: Yenilenebilir Enerji Volatilitesi
# Akarsu ve Rüzgar'ın yanına Güneş'i de ekliyoruz (Bulutluluk etkisi oynaklık yaratır).
yenilenebilir_cols = ['Akarsu', 'Rüzgar', 'Güneş']

# Fonksiyon: Gruplar için hareketli standart sapma çizdirme
def plot_grouped_volatility(df, columns, title):
    vol_data = df[columns].rolling(window=24).std()
    vol_data.plot(figsize=(12, 5), title=title)
    plt.ylabel("Standart Sapma (24s)")
    plt.grid(True, alpha=0.3)
    plt.show()

# Uygulama
plot_grouped_volatility(df_final, price_maliye_cols, "Fiyat ve Döviz Volatilitesi")
plot_grouped_volatility(df_final, fosil_cols, "Fosil Yakıt Üretim Volatilitesi")
plot_grouped_volatility(df_final, yenilenebilir_cols, "Yenilenebilir Enerji Üretim Volatilitesi")


#---------------------------
# KORELASYON
#---------------------------

# 1. Sayısal sütunları seçelim
numerical_cols = df_final.select_dtypes(include=[np.number]).columns

# 2. Spearman Korelasyon Matrisini hesaplayalım
spearman_corr = df_final[numerical_cols].corr(method='spearman')

# 3. Sadece PTF ile olan ilişkileri alıp sıralayalım
ptf_corr = spearman_corr[['PTF (TL/MWH)']].sort_values(by='PTF (TL/MWH)', ascending=False)

# 4. Görselleştirme
plt.figure(figsize=(8, 12))
sns.heatmap(ptf_corr, annot=True, cmap='RdYlGn', fmt=".2f", center=0)
plt.title("Değişkenlerin PTF ile Spearman Korelasyonu")
plt.show()



#---------------------------
# ÇOKLU BAĞLANTI (VIF)
#---------------------------

# 1. Sadece bağımsız değişkenleri seçelim (Bağımlı değişken PTF ve Tarih hariç)
X = df_final.drop(['PTF (TL/MWH)'], axis=1).select_dtypes(include=[np.number])

# 2. VIF için sabit (constant) eklenmesi önerilir (opsiyonel ama sağlıklı sonuç verir)
# Ancak VIF kütüphanesi genelde ham veriyle de çalışır.
vif_data = pd.DataFrame()
vif_data["Değişken"] = X.columns

# 3. Her değişken için VIF değerini hesapla
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data.sort_values(by="VIF", ascending=False))




# Korelasyonu düşük ama VIF'i çok yüksek olanları eleyerek testi tekrarlayalım
# Örn: Biyokütle, Jeotermal ve Akarsu'yu çıkarıyoruz
drop_list = ['Biyokütle', 'Jeotermal', 'Akarsu']
X_reduced = X.drop(columns=drop_list)

vif_reduced = pd.DataFrame()
vif_reduced["Değişken"] = X_reduced.columns
vif_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(len(X_reduced.columns))]

print("Gereksiz Değişkenler Elendikten Sonra VIF:")
print(vif_reduced.sort_values(by="VIF", ascending=False))



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Orijinal X verisi (tüm bağımsız değişkenler)

vif_scaled = pd.DataFrame()
vif_scaled["Değişken"] = X.columns
vif_scaled["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print("StandardScaler Sonrası VIF Değerleri:")
print(vif_scaled.sort_values(by="VIF", ascending=False))



# Durağan olmayan ve yüksek VIF verenlerin 1. derece farkını alalım
X_diff = X.diff().dropna()

vif_diff = pd.DataFrame()
vif_diff["Değişken"] = X_diff.columns
vif_diff["VIF"] = [variance_inflation_factor(X_diff.values, i) for i in range(len(X_diff.columns))]

print("Fark Alma (Differencing) Sonrası VIF Değerleri:")
print(vif_diff.sort_values(by="VIF", ascending=False))


#------------------------------------------------------------------------------------------------------------
# ZAMAN SERİSİ ANALİZİ
#------------------------------------------------------------------------------------------------------------

#---------------------------
# MEVSİMSELLİK
#---------------------------

# PTF verisini 24 saatlik periyotla (günlük döngü) ayrıştıralım
# Not: Veri setinde tarih indeksi olduğundan emin olmalısın
result = seasonal_decompose(df_final['PTF (TL/MWH)'], model='additive', period=24)


# Grafik ayarları
plt.rcParams['figure.figsize'] = (14, 12)
result.plot()
plt.suptitle('PTF (TL/MWH) 24 Saatlik Mevsimsel Ayrıştırma', fontsize=16, y=1.02)
plt.show()

result_short = seasonal_decompose(df_final['PTF (TL/MWH)'].tail(500), model='additive', period=24)
result_short.plot()
plt.show()

result_short = seasonal_decompose(df_final['PTF (TL/MWH)'].tail(500), model='additive', period=168)
result_short.plot()
plt.show()


#---------------------------
# ACF PACF
#---------------------------


# 1. Veriyi hazırla (NaN değerleri temizle)
ptf_series = df_final['PTF (TL/MWH)'].dropna()

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# 2. ACF Çizimi
plot_acf(ptf_series, lags=48, ax=axes[0])
axes[0].set_title('PTF ACF (48 Saatlik Gecikme)')

# 3. PACF Çizimi (Metod 'yw' olarak güncellendi)
plot_pacf(ptf_series, lags=48, ax=axes[1], method='yw')
axes[1].set_title('PTF PACF (48 Saatlik Gecikme)')

plt.tight_layout()
plt.show()


ptf_series = df_final['PTF (TL/MWH)'].dropna()

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# 2. ACF Çizimi
plot_acf(ptf_series, lags=170, ax=axes[0])
axes[0].set_title('PTF ACF (168  Saatlik Gecikme)')

# 3. PACF Çizimi (Metod 'yw' olarak güncellendi)
plot_pacf(ptf_series, lags=170, ax=axes[1], method='yw')
axes[1].set_title('PTF PACF (168 Saatlik Gecikme)')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

# Örn: Rüzgar Üretimi ile PTF arasındaki gecikmeli ilişki
# Rüzgar arttıktan kaç saat sonra fiyat düşüyor?
target = df_final['PTF (TL/MWH)'].dropna()
feature = df_final['Rüzgar'].dropna() # Sütun adını kendi df'ine göre güncelle

# Cross-correlation hesapla (ilk 24 saat için)
cross_corr = [target.corr(feature.shift(lag)) for lag in range(25)]

plt.figure(figsize=(10, 5))
plt.bar(range(25), cross_corr)
plt.title('Rüzgar Üretimi ve PTF Çapraz Korelasyonu (Lags)')
plt.xlabel('Gecikme (Saat)')
plt.ylabel('Korelasyon Katsayısı')
plt.show()




# 24 saatlik hareketli ortalama ve standart sapma
rolling_mean = df_final['PTF (TL/MWH)'].rolling(window=24).mean()
rolling_std = df_final['PTF (TL/MWH)'].rolling(window=24).std()

plt.figure(figsize=(14, 7))
plt.plot(df_final['PTF (TL/MWH)'], label='Orijinal PTF', alpha=0.3)
plt.plot(rolling_mean, label='24s Hareketli Ortalama', color='red')
plt.plot(rolling_std, label='24s Hareketli Oynaklık (Std)', color='black')
plt.title('PTF Hareketli İstatistik Analizi')
plt.legend()
plt.show()
#-------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Veriyi kopyala
df_heatmap = df_final.copy()

# 2. Sütun İsimlerini Kontrol Et (Debug için)
print("Sütunlar:", df_heatmap.columns.tolist())

# --- SAAT BİLGİSİNİ DÜZELTME OPERASYONU ---

# Senaryo A: Veride 'Saat' isminde ayrı bir sütun varsa onu kullan
# (Genelde string "00:00" veya integer 0,1,2.. formatında olabilir)
col_names = [c.lower() for c in df_heatmap.columns]

if 'saat' in col_names:
    # Gerçek sütun adını bul (Büyük/küçük harf duyarlı)
    saat_col = df_heatmap.columns[col_names.index('saat')]
    print(f"✅ 'Saat' sütunu bulundu: {saat_col}")

    # Eğer saat "00:00" formatındaysa sadece saati al, sayıysa direkt al
    try:
        df_heatmap['Hour'] = df_heatmap[saat_col].astype(str).str.split(':').str[0].astype(int)
    except:
        df_heatmap['Hour'] = df_heatmap[saat_col].astype(int)

# Senaryo B: Saat sütunu yoksa, İndeks veya Tarih sütunundan çekmeyi dene
else:
    print("⚠️ 'Saat' sütunu bulunamadı, Tarih sütunundan çekiliyor...")
    if 'Tarih' not in df_heatmap.columns:
        df_heatmap = df_heatmap.reset_index()

    # Tarih sütununu datetime yap
    date_col = df_heatmap.columns[0]  # İlk sütunu tarih varsayalım
    df_heatmap[date_col] = pd.to_datetime(df_heatmap[date_col])

    df_heatmap['Hour'] = df_heatmap[date_col].dt.hour

# --- DİĞER ZAMAN BİLGİLERİ ---
# Tarih sütunu (Month ve Day için)
if 'Tarih' in df_heatmap.columns:
    df_heatmap['Tarih'] = pd.to_datetime(df_heatmap['Tarih'])
    df_heatmap['Month'] = df_heatmap['Tarih'].dt.month
    df_heatmap['Day_of_Week'] = df_heatmap['Tarih'].dt.dayofweek
else:
    # Eğer reset_index yaptıysak
    date_col = df_heatmap.columns[0]
    df_heatmap[date_col] = pd.to_datetime(df_heatmap[date_col])
    df_heatmap['Month'] = df_heatmap[date_col].dt.month
    df_heatmap['Day_of_Week'] = df_heatmap[date_col].dt.dayofweek

# --- KONTROL ---
print(f"Benzersiz Saat Değerleri: {df_heatmap['Hour'].unique()}")
# Eğer burada hala sadece [0] görüyorsan, veride saat bilgisi hiç yok demektir!

# --- GRAFİKLERİ ÇİZ ---

# 1. Isı Haritası
pivot_table = df_heatmap.pivot_table(values='PTF (TL/MWH)', index='Hour', columns='Day_of_Week', aggfunc='mean')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=False)
plt.title('DÜZELTİLMİŞ PTF Isı Haritası (Saat vs Gün)')
plt.xlabel('Haftanın Günü (0=Pzt, 6=Pzr)')
plt.ylabel('Günün Saati (0-23)')
plt.show()

# 2. Kutu Grafikleri
fig, axes = plt.subplots(2, 1, figsize=(15, 12))

sns.boxplot(x='Hour', y='PTF (TL/MWH)', data=df_heatmap, ax=axes[0], palette="viridis")
axes[0].set_title('Saat Bazlı PTF Dağılımı (0-23 Arası Olmalı)')

sns.boxplot(x='Month', y='PTF (TL/MWH)', data=df_heatmap, ax=axes[1], palette="magma")
axes[1].set_title('Aylık PTF Dağılımı')

plt.tight_layout()
plt.show()

# =============================================================================
# -----------------------------------------------------------------------------
# ADIM 5: FEATURE ENGINEERING (ÖZELLİK MÜHENDİSLİĞİ) & SHIFT
# -----------------------------------------------------------------------------
# =============================================================================

# -----------------------------------------------------------------------------
# 1. SHIFT OPERASYONU (Hayati Önem Taşıyor!)

# -----------------------------------------------------------------------------
# Ötelenecek Üretim Verileri (Gerçekleşen oldukları için)
future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit',
               'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal', 'Biyokütle']

# Veri setinde hangileri varsa onları seçelim
cols_to_shift = [c for c in future_cols if c in df_final.columns]

print(f"⏳ Shift İşlemi: {len(cols_to_shift)} adet üretim değişkeni 24 saat ötelenecek...")

for col in cols_to_shift:
    # Mantık: Bugünün tahmini için DÜNÜN üretimini kullan.
    df_final[f'{col}_Lag24'] = df_final[col].shift(24)

    # Orijinal (Gelecek) sütunu sil ki model kopya çekmesin.
    df_final.drop(columns=[col], inplace=True)

print("✅ Shift tamamlandı. Model artık dürüst çalışacak.")

# -----------------------------------------------------------------------------
# 2. TARİH VE SAAT DÖNÜŞÜMLERİ (GÜNCELLENDİ)
# Neden: Saat 23->00 ve Pazar->Pazartesi geçişlerini modele öğretmek.
# -----------------------------------------------------------------------------
# Ay ve Gün Bilgisi
df_final['Month'] = df_final['Tarih'].dt.month
df_final['Day_of_Week'] = df_final['Tarih'].dt.dayofweek
df_final['Is_Weekend'] = df_final['Day_of_Week'].isin([5, 6]).astype(int)

# --- SAAT DÖNÜŞÜMÜ (Zaten Vardı) ---
if df_final['Saat'].dtype == 'O':
    df_final['Saat_Int'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int)
else:
    df_final['Saat_Int'] = df_final['Saat']

df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Saat_Int'] / 24)

# --- GÜN DÖNÜŞÜMÜ  ---

df_final['Day_Sin'] = np.sin(2 * np.pi * df_final['Day_of_Week'] / 7)
df_final['Day_Cos'] = np.cos(2 * np.pi * df_final['Day_of_Week'] / 7)


# -----------------------------------------------------------------------------
# 3. FİYAT HAFIZASI (LAG FEATURES)
# Neden: ACF Analizinde gördük, fiyat geçmişten etkilenmektedir.
# -----------------------------------------------------------------------------
target_col = 'PTF (TL/MWH)'

# Dün aynı saatte fiyat neydi? (Modelin en büyük yardımcısı budur)
df_final['PTF_Lag_24'] = df_final[target_col].shift(24)

# Geçen hafta aynı saatte fiyat neydi? (Haftalık döngüyü yakalar)
df_final['PTF_Lag_168'] = df_final[target_col].shift(168)

# Son 24 saatin ortalaması (Trend var mı?)
df_final['PTF_Roll_Mean_24'] = df_final[target_col].rolling(24).mean()


# -----------------------------------------------------------------------------
# 4. SNIPER ÖZELLİKLER (Overfitting Önleyici Akıllı Rasyolar)
# Neden: Kanıtladığımız en güçlü değişkenler.
# -----------------------------------------------------------------------------
print("🎯 Sniper Değişkenler Hesaplanıyor...")

# A. RELATIVE PRICE POSITION (En Güçlüsü)
# Fiyatın tarihsel ortalamasına göre konumu. Enflasyondan etkilenmez.
# Haftalık ortalamayı baz alıyoruz (168 saat).
df_final['PTF_Roll_Mean_168'] = df_final[target_col].rolling(168).mean()
# 0'a bölme hatası olmasın diye paydaya +1
df_final['Relative_Price_Pos'] = (df_final['PTF_Lag_24'] - df_final['PTF_Roll_Mean_168']) / (df_final['PTF_Roll_Mean_168'] + 1)

# B. NET YÜK (NET LOAD)
# Toplam Yükten Yenilenebilir Enerjiyi Çıkar -> Termikçilere kalan yük.
# Önce yenilenebilirleri topla (Shift edilmiş olanları!)
ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
existing_ren = [c for c in ren_cols if c in df_final.columns]
df_final['Total_Renewable_Lag24'] = df_final[existing_ren].sum(axis=1)

load_col = 'Yük Tahmin Planı (MWh)'
if load_col in df_final.columns:
    df_final['Net_Load'] = df_final[load_col] - df_final['Total_Renewable_Lag24']
else:
    # Yük yoksa negatif üretim olarak al
    df_final['Net_Load'] = -df_final['Total_Renewable_Lag24']

# C. THERMAL STRESS RATIO (Termik Stres)
# (Gaz + Kömür) / Toplam Yük. Sistem ne kadar zorda?
therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
existing_therm = [c for c in therm_cols if c in df_final.columns]
df_final['Total_Thermal_Lag24'] = df_final[existing_therm].sum(axis=1)

if load_col in df_final.columns:
    df_final['Thermal_Stress'] = df_final['Total_Thermal_Lag24'] / (df_final[load_col] + 1)
else:
    df_final['Thermal_Stress'] = 0

# D. PRICE MOMENTUM
# Haftalık değişim trendi (Artıyor mu azalıyor mu?)
df_final['Price_Momentum'] = df_final['PTF_Lag_24'] - df_final['PTF_Lag_168']

# E. VOLATILITY (Korku Endeksi)
# Son 24 saatteki fiyat oynaklığı (Standart Sapma).
# Bugünü görmemesi için shift(24) yapıyoruz.
df_final['Volatility'] = df_final[target_col].rolling(24).std().shift(24)


# -----------------------------------------------------------------------------
# 5. SON TEMİZLİK VE HAZIRLIK
# -----------------------------------------------------------------------------
# Shift ve Rolling(168) yaptığımız için ilk 1 hafta (168 satır) boşaldı.
# Onları siliyoruz.
print(f"🧹 Temizlik Öncesi Satır: {len(df_final)}")
df_final.dropna(inplace=True)
print(f"✅ Temizlik Sonrası Satır: {len(df_final)} (Modele Hazır)")

# Gereksiz sütunları (Modelin anlamadığı stringleri) atalım
# Tarih ve Saat'i modelden çıkarıyoruz ama grafik için saklayacağız (df_final'da kalsın).
model_cols = [c for c in df_final.columns if c not in ['Tarih', 'Saat', 'Zaman', 'Saat_Int']]

print(f"🧠 Modele Girecek Değişken Sayısı: {len(model_cols)}")
print(f"   Sniper'lar Dahil: Relative_Price_Pos, Net_Load, Thermal_Stress...")




#-------
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import holidays  # Tatiller için bu kütüphane şart: pip install holidays

# =============================================================================
# ADIM 5: FEATURE ENGINEERING (DÜZELTİLMİŞ VERSİYON)
# =============================================================================

# -----------------------------------------------------------------------------
# 0. TATİL DEĞİŞKENLERİ (EKSİKTİ, EKLENDİ)
# Türkiye takvimini ve dini bayramları çeker.
# -----------------------------------------------------------------------------
print("📅 Tatil Günleri İşleniyor...")
# Türkiye tatillerini al
tr_holidays = holidays.TR(years=[2023, 2024, 2025])

# 'Tarih' sütununun datetime olduğundan emin olalım
if 'Tarih' not in df_final.columns:
    df_final = df_final.reset_index()
    # İlk sütunu tarih varsay
    col = df_final.columns[0]
    df_final.rename(columns={col: 'Tarih'}, inplace=True)

df_final['Tarih'] = pd.to_datetime(df_final['Tarih'])

# Tatil mi? (0 veya 1)
df_final['Is_Holiday'] = df_final['Tarih'].apply(lambda x: 1 if x in tr_holidays else 0)

# Hafta sonu zaten vardı ama buraya da ekleyelim (Isı haritasından ders çıkardık)
df_final['Day_of_Week'] = df_final['Tarih'].dt.dayofweek
df_final['Is_Weekend'] = df_final['Day_of_Week'].isin([5, 6]).astype(int)


# -----------------------------------------------------------------------------
# 1. SHIFT OPERASYONU (DOĞRUYDU, AYNEN KORUNDU)
# -----------------------------------------------------------------------------
future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit',
               'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal', 'Biyokütle']

cols_to_shift = [c for c in future_cols if c in df_final.columns]

for col in cols_to_shift:
    # 24 Saat öteleme
    df_final[f'{col}_Lag24'] = df_final[col].shift(24)
    df_final.drop(columns=[col], inplace=True)

# -----------------------------------------------------------------------------
# 2. DURAĞANLAŞTIRMA / FARK ALMA (EKSİKTİ, EKLENDİ)
# Doğalgaz gibi trend içeren verilerin günlük değişimini alıyoruz.
# -----------------------------------------------------------------------------
trend_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24'] # Varsa Dolar'ı da ekle
cols_to_diff = [c for c in trend_cols if c in df_final.columns]

for col in cols_to_diff:
    # Hem Lag alınmış verinin farkını alıyoruz (Bugün - Dün)
    df_final[f'{col}_Diff'] = df_final[col].diff()
    # Orijinal Lag'li veriyi tutabilirsin veya silebilirsin (VIF durumuna göre)
    # Biz şimdilik tutalım, model seçsin.

# -----------------------------------------------------------------------------
# 3. SAAT VE GÜN DÖNÜŞÜMLERİ (DOĞRUYDU, KORUNDU)
# -----------------------------------------------------------------------------
if 'Saat' in df_final.columns:
    if df_final['Saat'].dtype == 'O':
        df_final['Saat_Int'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int)
    else:
        df_final['Saat_Int'] = df_final['Saat']
else:
    # Saat yoksa tarihten çek
    df_final['Saat_Int'] = df_final['Tarih'].dt.hour

# Trigonometrik Dönüşüm
df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Day_Sin'] = np.sin(2 * np.pi * df_final['Day_of_Week'] / 7)
df_final['Day_Cos'] = np.cos(2 * np.pi * df_final['Day_of_Week'] / 7)


# -----------------------------------------------------------------------------
# 4. FİYAT HAFIZASI VE SIZINTI ENGELLEME (DÜZELTİLDİ!)
# -----------------------------------------------------------------------------
target_col = 'PTF (TL/MWH)'

# Lag 24 ve 168 (Doğru)
df_final['PTF_Lag_24'] = df_final[target_col].shift(24)
df_final['PTF_Lag_168'] = df_final[target_col].shift(168)

# DÜZELTME: Rolling Mean Sızıntısı Engellendi
# Orijinal: df_final[target].rolling(24).mean() -> HATALI (Bugünü görür)
# Yeni: Lag_24 üzerinden ortalama alıyoruz. Yani "Dün bu saatten geriye 24 saat".
df_final['PTF_Roll_Mean_24'] = df_final['PTF_Lag_24'].rolling(24).mean()
df_final['PTF_Roll_Std_24'] = df_final['PTF_Lag_24'].rolling(24).std()


# -----------------------------------------------------------------------------
# 5. SNIPER ÖZELLİKLER (DOĞRUYDU, KORUNDU)
# -----------------------------------------------------------------------------
# A. Relative Price Position (Güvenli, çünkü Lag_24 kullanıyor)
df_final['PTF_Roll_Mean_168'] = df_final['PTF_Lag_24'].rolling(168).mean()
df_final['Relative_Price_Pos'] = (df_final['PTF_Lag_24'] - df_final['PTF_Roll_Mean_168']) / (df_final['PTF_Roll_Mean_168'] + 1)

# B. Net Load (Yenilenebilir Toplamı)
ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
existing_ren = [c for c in ren_cols if c in df_final.columns]
df_final['Total_Renewable_Lag24'] = df_final[existing_ren].sum(axis=1)

load_col = 'Yük Tahmin Planı (MWh)'
if load_col in df_final.columns:
    df_final['Net_Load'] = df_final[load_col] - df_final['Total_Renewable_Lag24']
else:
    df_final['Net_Load'] = -df_final['Total_Renewable_Lag24']

# C. Thermal Stress Ratio
therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
existing_therm = [c for c in therm_cols if c in df_final.columns]
df_final['Total_Thermal_Lag24'] = df_final[existing_therm].sum(axis=1)

if load_col in df_final.columns:
    df_final['Thermal_Stress'] = df_final['Total_Thermal_Lag24'] / (df_final[load_col] + 1)

# D. Momentum
df_final['Price_Momentum'] = df_final['PTF_Lag_24'] - df_final['PTF_Lag_168']


# -----------------------------------------------------------------------------
# 6. TEMİZLİK
# -----------------------------------------------------------------------------
print(f"🧹 Temizlik Öncesi: {len(df_final)}")
df_final.dropna(inplace=True)
print(f"✅ Temizlik Sonrası: {len(df_final)}")

# Modele girmeyecek sütunları belirle (Tarih, Saat, vs.)
exclude_cols = ['Tarih', 'Saat', 'Zaman', 'Saat_Int', 'PTF (TL/MWH)'] # Hedef değişkeni de X'ten ayırırken kullanacağız
feature_cols = [c for c in df_final.columns if c not in exclude_cols]

print(f"🚀 Hazır Özellik Sayısı: {len(feature_cols)}")
print(feature_cols)




# =============================================================================
# ADIM 6: MODELLEME
# ==============================================


# -----------------------------------------------------------------------------
# 1. X (ÖZELLİKLER) ve y (HEDEF) AYRIMI
# -----------------------------------------------------------------------------

# Hedef Değişkenimiz
target_col = 'PTF (TL/MWH)'

# Modelin görmemesi gereken (Drop Listesi) sütunlar
# Not: 'Yük Tahmin Planı (MWh)' şimdilik kalıyor.
drop_cols = [
    'Tarih',        # Datetime formatı, model işlemez
    'Zaman',        # Datetime formatı, model işlemez
    'Saat',         # String/Object formatı veya gereksiz tekrar
    'Saat_Int',     # Hour_Sin/Cos varken bazen gereksiz olabilir ama sayısal olduğu için kalabilir.
    target_col      # HEDEF DEĞİŞKEN (Sızıntıyı önlemek için X'ten atıyoruz)
]

# Sadece veri setinde mevcut olanları drop listesine ekle (Hata almamak için)
existing_drop_cols = [c for c in drop_cols if c in df_final.columns]

# X Matrisi (Girdiler)
X = df_final.drop(columns=existing_drop_cols)

# y Vektörü (Çıktı / Hedef)
y = df_final[target_col]

# Tarihleri Görselleştirme İçin Sakla (Senin Kodun - Dinamik Hali)
dates = df_final['Tarih']

print(f"🚫 Drop Edilen Sütunlar: {existing_drop_cols}")
print(f"✅ X Matrisi Boyutu: {X.shape}")
print(f"🎯 y Matrisi Boyutu: {y.shape}")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 2. ZAMAN SERİSİ BÖLÜMLEME (TRAIN / TEST SPLIT) - TARİH BAZLI
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Strateji: Kesin tarih aralıklarına göre eğitim ve test setlerini ayırıyoruz.
# Train: 01.01.2024 - 31.10.2025 (Öğrenme Dönemi)
# Test:  01.11.2025 - 30.11.2025 (Sınav Dönemi - Sadece Kasım Ayı)

# Tarih sınırlarını tanımlayalım (Pandas kıyaslaması için YYYY-MM-DD formatı en iyisidir)
train_end_date = '2025-10-31'
test_start_date = '2025-11-01'
test_end_date = '2025-11-30'

# Maskeleme (Filtreleme) Oluşturma
# X ve y matrislerinde 'Tarih' sütunu olmadığı için, dışarıdaki 'dates' değişkenini referans alıyoruz.
train_mask = (dates >= '2024-01-01') & (dates <= train_end_date)
test_mask  = (dates >= test_start_date) & (dates <= test_end_date)

# Veriyi Bölme (.loc kullanarak)
X_train = X.loc[train_mask]
X_test  = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test  = y.loc[test_mask]

# Tarihleri de ayıralım (Grafik ve analizler için lazım olacak)
dates_train = dates.loc[train_mask]
dates_test  = dates.loc[test_mask]

# KONTROL (İstediğin net tarih aralıklarını teyit edelim)
print("-" * 50)
print(f"📉 Eğitim Seti (Train): {len(X_train)} satır")
print(f"   Aralık: {dates_train.min().date()}  --->  {dates_train.max().date()}")
print("-" * 50)
print(f"📈 Test Seti (Test):    {len(X_test)} satır")
print(f"   Aralık: {dates_test.min().date()}  --->  {dates_test.max().date()}")
print("-" * 50)

# Güvenlik Kontrolü: Test seti boş mu? (Tarih formatı hatası varsa uyarması için)
if len(X_test) == 0:
    raise ValueError("⚠️ HATA: Test seti boş geldi! Tarih formatlarını veya veri aralığını kontrol et.")

# -----------------------------------------------------------------------------
# 3. REFERANS NOKTASI (BENCHMARK - NAIVE FORECAST)
# -----------------------------------------------------------------------------
# "Yarınki fiyat, bugünkü fiyattır" (veya Lag 168 - geçen haftadır)
# Biz Lag_24 (Dünkü fiyat) üzerinden Naive Forecast yapalım.
# Test setindeki 'PTF_Lag_24' sütununu tahmin olarak kabul ediyoruz.

if 'PTF_Lag_24' in X_test.columns:
    naive_pred = X_test['PTF_Lag_24']
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    naive_mae = mean_absolute_error(y_test, naive_pred)

    print(f"🛑 Benchmark (Naive) RMSE: {naive_rmse:.2f} TL")
    print(f"🛑 Benchmark (Naive) MAE:  {naive_mae:.2f} TL")
    print("   -> Hedefimiz bu hataların altına düşmek!")
else:
    print("⚠️ PTF_Lag_24 bulunamadı, Benchmark atlanıyor.")


# -----------------------------------------------------------------------------
# 4. HİPERPARAMETRE OPTİMİZASYONU (TUNING) - RandomizedSearch
# -----------------------------------------------------------------------------
print("\n⚙️ Hiperparametre Optimizasyonu Başlıyor... (Bu biraz sürebilir)")

# Parametre Uzayı (Arama Yapılacak Ayarlar)
param_dist = {
    'n_estimators': [500, 1000, 1500],        # Ağaç sayısı
    'learning_rate': [0.01, 0.05, 0.1],       # Öğrenme hızı (Küçük olması iyidir ama yavaştır)
    'max_depth': [3, 5, 7, 9],                # Ağaç derinliği (Çok derin = Overfitting riski)
    'subsample': [0.7, 0.8, 0.9],             # Her ağaç için verinin ne kadarını kullansın
    'colsample_bytree': [0.7, 0.8, 0.9],      # Her ağaç için sütunların ne kadarını kullansın
    'objective': ['reg:squarederror']         # Regresyon görevi
}

# Base Model
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1) # n_jobs=-1 tüm işlemciyi kullanır

# Zaman Serisi Cross-Validation (Shuffle yok!)
tscv = TimeSeriesSplit(n_splits=3)

# Randomized Search (Grid Search'ten daha hızlıdır)
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=10,  # 10 farklı kombinasyon dene (Hız için düşük tuttuk, artırabilirsin)
    scoring='neg_root_mean_squared_error',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Aramayı Başlat (Sadece Train seti üzerinde!)
random_search.fit(X_train, y_train)

print(f"\n🏆 En İyi Parametreler: {random_search.best_params_}")



# -----------------------------------------------------------------------------
# 5. FİNAL MODELİN EĞİTİLMESİ (BEST MODEL)
# -----------------------------------------------------------------------------
print("\n🦾 Final Model Eğitiliyor...")

# En iyi parametrelerle modeli al
best_model = random_search.best_estimator_

# Modeli tekrar eğit (Opsiyonel: Early Stopping ile)
# Early Stopping: Test setinde hata artmaya başlarsa eğitimi durdur.
eval_set = [(X_train, y_train), (X_test, y_test)]
best_model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=False  # Her satırı yazdırmasın
)



# -----------------------------------------------------------------------------
# 6. TAHMİN VE PERFORMANS ÖLÇÜMÜ (METRICS)
# -----------------------------------------------------------------------------
y_pred = best_model.predict(X_test)

# Negatif tahminleri engelle (Fiyat eksi olamaz - istisnalar hariç)
y_pred = np.maximum(y_pred, 0)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# MAPE Hesaplama (Sıfıra bölme hatasını engellemek için)
mask = y_test != 0
mape = (np.abs((y_test - y_pred) / y_test)[mask]).mean() * 100

print("\n" + "="*30)
print("📊 FİNAL MODEL SONUÇLARI")
print("="*30)
print(f"✅ Model RMSE: {rmse:.2f} TL (Hedef: < {naive_rmse:.2f})")
print(f"✅ Model MAE:  {mae:.2f} TL")
print(f"✅ Model MAPE: %{mape:.2f}")

improvement = ((naive_rmse - rmse) / naive_rmse) * 100
print(f"🚀 Naive Modele Göre İyileşme: %{improvement:.2f}")



# -----------------------------------------------------------------------------
# 7. GÖRSELLEŞTİRME (VISUALIZATION)
# -----------------------------------------------------------------------------
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

# Özellik Önem Düzeyi (Feature Importance)
plt.figure(figsize=(10, 8))
# En önemli 20 özelliği çizdir
sorted_idx = best_model.feature_importances_.argsort()[-20:]
plt.barh(X.columns[sorted_idx], best_model.feature_importances_[sorted_idx])
plt.title("XGBoost: En Önemli Değişkenler (Feature Importance)")
plt.xlabel("Önem Düzeyi")
plt.show()