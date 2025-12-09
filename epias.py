import pandas as pd
import yfinance as yf
import warnings
import os

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

print("🚀 Dolar Kuru Kurtarma Operasyonu Başlıyor...")

# =============================================================================
# 1. ADIM: DOSYAYI OKU (Separator Kontrollü)
# =============================================================================
filename = "EPIAS_Project_Dataset.csv"  # Senin dosyanın adı

if not os.path.exists(filename):
    print(f"❌ HATA: '{filename}' bulunamadı! Dosya adını kontrol et.")
    exit()

# Önce virgül (standart) ile dene, olmazsa noktalı virgül ile dene
try:
    df = pd.read_csv(filename, sep=",")
    if len(df.columns) < 2:  # Eğer tek sütun okuduysa ayraç yanlıştır
        df = pd.read_csv(filename, sep=";")
    print(f"📂 Dosya Yüklendi: {len(df)} satır.")
except Exception as e:
    print(f"❌ Okuma Hatası: {e}")
    exit()

# =============================================================================
# 2. ADIM: TARİH FORMATINI 'ASKERİ NİZAM'A SOK (En Kritik Yer!)
# =============================================================================
print("🧹 Tarihler temizleniyor...")

# Senin verindeki tarihi datetime yap -> Saatleri sil -> Timezone varsa sil
df['Tarih'] = pd.to_datetime(df['Tarih'], dayfirst=True, errors='coerce')
df['Tarih'] = df['Tarih'].dt.normalize()  # Saatleri 00:00 yapar
df['Tarih'] = df['Tarih'].dt.tz_localize(None)  # Timezone bilgisini siler (Çok Önemli!)

# Bozuk tarih varsa uyar
if df['Tarih'].isnull().sum() > 0:
    print(f"⚠️ UYARI: {df['Tarih'].isnull().sum()} satırda tarih okunamadı!")
    df = df.dropna(subset=['Tarih'])  # Tarihsiz satırları at

# =============================================================================
# 3. ADIM: DOLAR KURUNU ÇEK VE AYNI FORMATA GETİR
# =============================================================================
print("💵 Yahoo Finance'ten veri çekiliyor...")

start_date = df['Tarih'].min()
end_date = df['Tarih'].max()

# Veriyi indir
try:
    usd_data = yf.download('TRY=X', start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
except Exception as e:
    print(f"❌ Yahoo Finance Hatası: {e}")
    exit()

# Yahoo verisini düzenle
usd_data = usd_data['Close'].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']

# Yahoo tarihini de senin verinle AYNI formata getir
usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih'])
usd_data['Tarih'] = usd_data['Tarih'].dt.normalize()
usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)  # Timezone sil (Eşleşme için şart)

# =============================================================================
# 4. ADIM: HAFTA SONU BOŞLUKLARINI DOLDUR
# =============================================================================
# Tarih iskeleti oluştur (Her günü kapsasın)
all_dates = pd.DataFrame({'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')})
all_dates['Tarih'] = all_dates['Tarih'].dt.normalize().dt.tz_localize(None)

# Dolar verisini iskelete oturt
usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')

# Cuma kurunu hafta sonuna yay (Forward Fill)
usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()

# =============================================================================
# 5. ADIM: BİRLEŞTİR (Left Join)
# =============================================================================
print("🔗 Veriler birleştiriliyor...")

# Eğer dosyada zaten bozuk bir Dolar sütunu varsa sil
if 'Dolar_Kuru' in df.columns:
    df.drop(columns=['Dolar_Kuru'], inplace=True)

# Birleştirme
df_final = pd.merge(df, usd_data, on='Tarih', how='left')

# Kontrol
nan_sayisi = df_final['Dolar_Kuru'].isnull().sum()

if nan_sayisi == 0:
    print(f"✅ MÜKEMMEL! Dolar kuru tüm satırlara ({len(df_final)}) başarıyla işlendi.")
    output_file = "EPIAS_Cleaned_With_USD.csv"
    df_final.to_csv(output_file, index=False)
    print(f"💾 Kaydedildi: {output_file}")

    # İlk 5 satırı göster
    print("\n--- ÖRNEK VERİ ---")
    print(df_final[['Tarih', 'Dolar_Kuru']].head())
else:
    print(f"❌ HATA: Hala {nan_sayisi} satırda Dolar yok! Tarih formatlarına tekrar bakmamız lazım.")
    print("Senin Verin Örnek Tarih:", df['Tarih'].iloc[0])
    print("Dolar Verisi Örnek Tarih:", usd_data['Tarih'].iloc[0])