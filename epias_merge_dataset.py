from statistics import quantiles
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
from sklearn.exceptions import ConvergenceWarning
import datetime
import locale
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
df_final = pd.read_csv("data_s/güncel_set.csv")
df_final.head(20)
# ---------------------------
# DEĞİŞKEN TİPİ DÜZELTME
# ---------------------------
def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    return float(x)

object_to_float = [col for col in df_final.columns if col not in ['Tarih', 'Saat']]
for col in object_to_float:
    df_final[col] = df_final[col].apply(clean_currency)

# ---------------------------
# TARİH NORMALİZE (EN KRİTİK KISIM)
# ---------------------------
# Formatı açıkça belirt (En sağlam ve hızlı yöntem budur)
df_final['Tarih'] = pd.to_datetime(df_final['Tarih'], format='%d.%m.%Y').dt.normalize()

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
df_final.head(100)
# ---------------------------
# BOTAS DOĞALGAZ FİYATI
# ---------------------------
sinir_tarih = pd.Timestamp('2025-07-01')

df_final['dogalgaz_fiyatlari_Mwh'] = np.where(
    df_final['Tarih'] <= sinir_tarih,
    1127.82,
    1409.77
)
df_final.head(100)
# ---------------------------
# GEREKSİZ SÜTUNLAR
# ---------------------------
drop_list = [
    'PTF (USD/MWh)', 'PTF (EUR/MWh)',
    'Toplam(MWh)',
    'Nafta', 'Fueloil',
    'Taş Kömür', 'Diğer'
]

existing_drop = [col for col in drop_list if col in df_final.columns]
df_final.drop(columns=existing_drop, inplace=True)

# ---------------------------
# KAYIT
# ---------------------------
df_final.to_csv('EPIAS_yeni.csv', index=False)

# ---------------------------
# KONTROL
# ---------------------------
print(df_final[['Tarih', 'Saat', 'Dolar_Kuru']].head(10))


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
# Aykırı gözlem var mı incelemesi
#---------------------------

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


#  Aykırı değerlerin bulunduğu değişkenleri bulmak
outlier_var = []
for col in num_cols:
    if check_outlier(df_final, col):
        outlier_var.append(col)
    if col != "SalePrice":
        print(col, check_outlier(df_final, col))

for var in outlier_var:
    sns.boxplot(data=df_final, x=df_final[var])
    plt.show()

#---------------------------
# FUTURE ENGİNEERİNG
#---------------------------
#  Outlier values
for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df_final,col)

# Kontrol edelim!!!
for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df_final, col))




df_final['TOPLAM_URETIM'] = (
    df_final['Doğalgaz'] +
    df_final['Linyit'] +
    df_final['İthal Kömür'] +
    df_final['Jeotermal'] +
    df_final['Barajlı'] +
    df_final['Akarsu'] +
    df_final['Rüzgar'] +
    df_final['Güneş'] +
    df_final['Biyokütle']
)
df_final.info()
df_final['YENILENEBILIR_TOPLAM'] = (
    df_final['Rüzgar'] +
    df_final['Güneş'] +
    df_final['Barajlı'] +
    df_final['Akarsu'] +
    df_final['Jeotermal'] +
    df_final['Biyokütle']
)
df_final['NET_YUK'] = df_final['Yük Tahmin Planı (MWh)'] - df_final['YENILENEBILIR_TOPLAM']


df_final['TOPLAM_TERMIK_URETIM'] = (
    df_final['Doğalgaz'] +
    df_final['Linyit'] +
    df_final['İthal Kömür']
)

df_final['YENILENEBILIR_ORANI'] = df_final['YENILENEBILIR_TOPLAM'] / df_final['TOPLAM_URETIM']

df_final['PTF_lag_1'] = df_final['PTF (TL/MWH)'].shift(1)
df_final['PTF_lag_24'] = df_final['PTF (TL/MWH)'].shift(24)
df_final['PTF_lag_168'] = df_final['PTF (TL/MWH)'].shift(168)

df_final['PTF_roll_mean_24'] = df_final['PTF (TL/MWH)'].rolling(24).mean()
df_final['PTF_roll_std_24'] = df_final['PTF (TL/MWH)'].rolling(24).std()

df_final.head(200)




#Lag ve rolling sonrası ilk satırlar boş olur:

####df_final = df_final.dropna().reset_index(drop=True)
## Tüm verileri kontrol ettikten sonra silinecek !!!!


# 1. DOĞALGAZ ETKİSİ (Gas Impact)
# Mantık: O saatte ne kadar çok gaz yakılıyorsa, gazın birim fiyatı o kadar önem kazanır.
# dogalgaz_fiyatlari_Mwh genelde TL bazlıdır (BOTAŞ Tarifesi).
# Eğer bu veri zaten TL ise Dolar ile çarpmaya gerek yoktur.

df_final['Gas_Impact'] = df_final['Doğalgaz'] * df_final['dogalgaz_fiyatlari_Mwh']


# 2. İTHAL KÖMÜR ETKİSİ (Coal Impact / FX Impact)
# Mantık: İthal kömür yurt dışından Dolar ile alınır.
# Elimizde kömürün ton fiyatı (API2 endeksi) yok ama Dolar Kuru var.
# İthal Kömür üretimi arttıkça, Dolar kurunun maliyet üzerindeki baskısı artar.

df_final['Coal_Impact'] = df_final['İthal Kömür'] * df_final['Dolar_Kuru']




# 1. AYARLAR VE TATİL LİSTESİ
# ---------------------------------------------------------
# Tarih formatı ayarı (Senin kodundan alındı)
try:
    locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'Turkish_Turkey.1254')
    except locale.Error:
        pass


def get_holiday_dates_2025():
    """
    Senin sağladığın listeden sadece tarih objelerini (datetime.date)
    bir liste olarak döndürür.
    """
    holidays = [
        (datetime.date(2025, 1, 1), "Yılbaşı"),

        # Ramazan Bayramı
        (datetime.date(2025, 3, 29), "Ramazan Bayramı Arifesi"),
        (datetime.date(2025, 3, 30), "Ramazan Bayramı 1. Gün"),
        (datetime.date(2025, 3, 31), "Ramazan Bayramı 2. Gün"),
        (datetime.date(2025, 4, 1), "Ramazan Bayramı 3. Gün"),

        # Ulusal Bayramlar
        (datetime.date(2025, 4, 23), "Ulusal Egemenlik ve Çocuk Bayramı"),
        (datetime.date(2025, 5, 1), "Emek ve Dayanışma Günü"),
        (datetime.date(2025, 5, 19), "Atatürk'ü Anma, Gençlik ve Spor Bayramı"),

        # Kurban Bayramı
        (datetime.date(2025, 6, 5), "Kurban Bayramı Arifesi"),
        (datetime.date(2025, 6, 6), "Kurban Bayramı 1. Gün"),
        (datetime.date(2025, 6, 7), "Kurban Bayramı 2. Gün"),
        (datetime.date(2025, 6, 8), "Kurban Bayramı 3. Gün"),
        (datetime.date(2025, 6, 9), "Kurban Bayramı 4. Gün"),

        # Diğer Resmi Tatiller
        (datetime.date(2025, 7, 15), "Demokrasi ve Milli Birlik Günü"),
        (datetime.date(2025, 8, 30), "Zafer Bayramı"),
        (datetime.date(2025, 10, 28), "Cumhuriyet Bayramı Arifesi"),
        (datetime.date(2025, 10, 29), "Cumhuriyet Bayramı"),
    ]

    # Bize sadece tarihlerin olduğu bir liste lazım
    return [date_obj for date_obj, name in holidays]


# 2. VERİ SETİ HAZIRLIĞI (SİMÜLASYON)
# ---------------------------------------------------------
# Gerçek verin olmadığı için 2025 yılı için saatlik boş bir veri seti oluşturuyorum.
# Sen kendi projende: df_final = pd.read_csv("verin.csv") yapacaksın.




# 3. FEATURE ENGINEERING (ÖZELLİK MÜHENDİSLİĞİ)
# ---------------------------------------------------------
# Önce veriyi string yap, sonra ':' işaretinden böl, ilk parçasını al ve sayıya çevir.
df_final['Saat'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int)

# Şimdi orijinal kodunu tekrar çalıştırabilirsin
df_final['Is_Peak_Hour'] = df_final['Saat'].apply(lambda x: 1 if 8 <= x <= 20 else 0)

# Kontrol
print(df_final[['Saat', 'Is_Peak_Hour']].head())
# A) Tatil Listesini Alalım
resmi_tatil_gunleri = get_holiday_dates_2025()

# B) Is_Weekend (Hafta Sonu Mu?)
# 5: Cumartesi, 6: Pazar
df_final['Is_Weekend'] = pd.to_datetime(df_final['Tarih']).dt.dayofweek.isin([5, 6]).astype(int)

# C) Is_Official_Holiday (Resmi Tatil Listesinde Var Mı?)
# Tarih sütunu ile tatil listesini karşılaştırır
df_final['Is_Official_Holiday'] = df_final['Tarih'].isin(resmi_tatil_gunleri).astype(int)

# D) Is_Holiday (Genel Tatil Durumu)
# Hafta sonu VEYA Resmi tatil ise 1 olsun.
# (Çünkü model için Pazar günü ile Bayram günü etkisi benzerdir: Talep düşer)
df_final['Is_Holiday'] = (df_final['Is_Weekend'] | df_final['Is_Official_Holiday']).astype(int)

# E) Is_Peak_Hour (Puant Saati Mi?)
# Sabah 08:00 ile Akşam 20:00 (dahil) arası
df_final['Is_Peak_Hour'] = df_final['Saat'].apply(lambda x: 1 if 8 <= x <= 20 else 0)

# F) Is_Business_Day (İş Günü Mü?)
# Tatil değilse iş günüdür. (Ters mantık)
df_final['Is_Business_Day'] = 1 - df_final['Is_Holiday']

df_final.head(200)