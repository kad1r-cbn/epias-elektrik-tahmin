import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import glob
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#---------------------------
# VERİ SETİ OKUMA
#---------------------------


ptf_df = pd.read_csv("data_s/Piyasa_Takas_Fiyati(PTF).csv", sep=";")
yuk_df = pd.read_csv("data_s/Yuk_Tahmin_Plani.csv", sep=";")
kgup_files = [
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-01012025-01042025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-02042025-02072025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03072025-03102025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03102025-30112025.csv"
]
kgup_dfs = []
for file in kgup_files:
    df = pd.read_csv(file,sep=";")
    kgup_dfs.append(df)
kgup_df = pd.concat(kgup_dfs)
kgup_df = kgup_df.drop_duplicates(subset=['Tarih', 'Saat']).reset_index(drop=True)

#---------------------------
# Tarih Formatı Değiştirme ve Merge İşlemleri
#---------------------------

# 1.Merge İşlemi
merge_1 = pd.merge(ptf_df,yuk_df, on=["Tarih","Saat"] ,how="inner" )

# 1.Tarih Format Değişimi
merge_1['Tarih'] = pd.to_datetime(merge_1['Tarih'], dayfirst=True, errors='coerce').dt.normalize()
kgup_df['Tarih'] = pd.to_datetime(kgup_df['Tarih'], dayfirst=True, errors='coerce').dt.normalize()

# Saat Formatı Eşitleme
merge_1['Saat'] = merge_1['Saat'].astype(str).str.strip().str[:5]
kgup_df['Saat'] = kgup_df['Saat'].astype(str).str.strip().str[:5]

# 2.Merge İşlemi
df_final = pd.merge(merge_1, kgup_df, on=["Tarih","Saat"] ,how="inner" ).reset_index(drop=True)
df_final = df_final.sort_values(by=["Tarih", "Saat"]).reset_index(drop=True)



#---------------------------
# Değişken Tiplerini Düzeltme
#---------------------------
def clean_currency(x):
    if isinstance(x, str):
        # 1. Önce binlik ayracı olan NOKTALARI tamamen sil
        x = x.replace('.', '')
        # 2. Sonra ondalık ayracı olan VİRGÜLLERİ noktaya çevir
        x = x.replace(',', '.')
    return float(x)
obcejt_to_str = [col for col in df_final.columns if col not in ['Tarih', 'Saat']]
for col in obcejt_to_str:
    df_final[col] = df_final[col].apply(clean_currency)

# df_final.to_csv('EPIAS_Project_Dataset.csv', index=False)

#---------------------------
# Dolar Kurunu Ekleme(Yahoo)
#---------------------------
start_date = df_final['Tarih'].min()
end_date = df_final['Tarih'].max()

usd_data = yf.download('TRY=X', start=start_date, end=end_date + pd.Timedelta(days=5))
usd_data = usd_data['Close'].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']

usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize()
usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)



# Datelerdeki boşluk dolar değerlerini doldurduk
all_dates = pd.DataFrame({'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')})
all_dates['Tarih'] = all_dates['Tarih'].dt.normalize().dt.tz_localize(None)

usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')

usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()

# Ana veriye ekle
df_final = pd.merge(df_final, usd_data, on='Tarih', how='left')

#---------------------------
# Gereksiz Değişkenleri Veri Setinden Atma
#---------------------------
drop_list = [
    'PTF (USD/MWh)', 'PTF (EUR/MWh)',
    'Toplam(MWh)',
    'Nafta', 'Fueloil',
    'Taş Kömür', 'Diğer'
]
existing_drop = [col for col in drop_list if col in df_final.columns]
if existing_drop:
    df_final.drop(columns=existing_drop, inplace=True)

