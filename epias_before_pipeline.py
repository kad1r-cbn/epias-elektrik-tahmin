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






