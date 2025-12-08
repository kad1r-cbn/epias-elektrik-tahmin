import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import glob

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

#ptf_dataset
ptf_df = pd.read_csv("veri_setleri/Piyasa_Takas_Fiyati(PTF).csv", sep=";")
ptf_df['Tarih'] = pd.to_datetime(ptf_df['Tarih'], dayfirst=True, errors='coerce')
ptf_df.head()

ptf_df.shape
#yuk_tahmin_dataset
yuk_df = pd.read_csv("veri_setleri/Yuk_Tahmin_Plani.csv", sep=";")
yuk_df['Tarih'] = pd.to_datetime(yuk_df['Tarih'], dayfirst=True, errors='coerce')
yuk_df.shape


# kgüp_datasets
kgup_files = [
    "veri_setleri/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-01012025-01042025.csv",
    "veri_setleri/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-02042025-02072025.csv",
    "veri_setleri/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03072025-03102025.csv",
    "veri_setleri/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03102025-30112025.csv"
]
kgup_dfs = []
for file in kgup_files:
    df = pd.read_csv(file,sep=";")
    kgup_dfs.append(df)

kgup_df = pd.concat(kgup_dfs)
kgup_df.shape

#merge

merge_1 = pd.merge(ptf_df,yuk_df, on=["Tarih","Saat"] ,how="inner" )
merge_1.head()
merge_1.shape



kgup_df = kgup_df.drop_duplicates(subset=['Tarih', 'Saat']).reset_index(drop=True)
kgup_df['Tarih'] = pd.to_datetime(kgup_df['Tarih'], dayfirst=True, errors='coerce')
kgup_df.head()
kgup_df.shape

#son düzeltme
merge_1['Tarih'] = pd.to_datetime(merge_1['Tarih'], dayfirst=True, errors='coerce').dt.normalize()
kgup_df['Tarih'] = pd.to_datetime(kgup_df['Tarih'], dayfirst=True, errors='coerce').dt.normalize()

#saat formatını da eşitledik
merge_1['Saat'] = merge_1['Saat'].astype(str).str.strip().str[:5]
kgup_df['Saat'] = kgup_df['Saat'].astype(str).str.strip().str[:5]


df_final = pd.merge(merge_1, kgup_df, on=["Tarih","Saat"] ,how="inner" ).reset_index(drop=True)
df_final.info()
print(f"\n✅ DÜZELTİLMİŞ SONUÇ: {len(df_final)} satır birleşti.")
df_final = df_final.sort_values(by=["Tarih", "Saat"]).reset_index(drop=True)
df_final.head()

df_final.isnull().sum()

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

df_final.head()
df_final.info()




df_final.to_csv('EPIAS_Project_Dataset.csv', index=False)

df.info()
df_final.shape
df_final.head()
