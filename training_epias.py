# =============================================================================
# TRAINING.PY - APP.PY İLE %100 UYUMLU FİNAL MOTOR
# =============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import joblib
import warnings
import os
import holidays

# Ayarlar
warnings.filterwarnings("ignore")
print("🚀 EĞİTİM MOTORU BAŞLATILIYOR (APP UYUMLU VERSİYON)...\n")

# -----------------------------------------------------------------------------
# 1. VERİ YÜKLEME VE TEMİZLİK
# -----------------------------------------------------------------------------
file_path = os.path.join('data_s', 'data_set_ex.xlsx')

try:
    df = pd.read_excel(file_path)
    print(f"✅ Veri Seti Yüklendi: {file_path}")
except:
    print(f"⚠️ Excel bulunamadı, CSV aranıyor...")
    df = pd.read_csv('data_s/data_set_ex.xlsx - Gercek Zamanli Uretim.csv')

# Sütun İsim Temizliği
df.columns = [col.strip() for col in df.columns]
target_col = 'PTF (TL/MWH)'
load_col = 'Yük Tahmin Planı (MWh)'


# Para Birimi Temizliği
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('.', '').replace(',', '.'))
    return float(x)


if df[target_col].dtype == 'O': df[target_col] = df[target_col].apply(clean_currency)
if df[load_col].dtype == 'O': df[load_col] = df[load_col].apply(clean_currency)

df['Tarih'] = pd.to_datetime(df['Tarih'], errors='coerce')

# Negatifleri Sıfırla
prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit', 'Akarsu', 'İthal Kömür']
for col in prod_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] < 0, col] = 0

# Outlier Interpolation
df.loc[df[target_col] > 90000, target_col] = np.nan
df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING (APP.PY İLE BİREBİR AYNI)
# -----------------------------------------------------------------------------
print("🛠️ Özellikler İşleniyor (Feature Engineering)...")
df_final = df.copy()

# A. Tarih/Saat Değişkenleri
if 'Saat' in df_final.columns:
    if df_final['Saat'].dtype == 'O':
        df_final['Saat_Int'] = df_final['Saat'].astype(str).str.split(':').str[0].astype(int)
    else:
        df_final['Saat_Int'] = df_final['Saat']
else:
    df_final['Saat_Int'] = df_final['Tarih'].dt.hour

# Tatil (Holidays)
tr_holidays = holidays.TR(years=[2023, 2024, 2025, 2026])
df_final['Is_Holiday'] = df_final['Tarih'].apply(lambda x: 1 if x in tr_holidays else 0)

df_final['Month'] = df_final['Tarih'].dt.month  # Modelde kullanılmasa da durabilir
df_final['Day_of_Week'] = df_final['Tarih'].dt.dayofweek
df_final['Is_Weekend'] = df_final['Day_of_Week'].isin([5, 6]).astype(int)

# Sin/Cos Dönüşümleri
df_final['Hour_Sin'] = np.sin(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Hour_Cos'] = np.cos(2 * np.pi * df_final['Saat_Int'] / 24)
df_final['Day_Sin'] = np.sin(2 * np.pi * df_final['Day_of_Week'] / 7)
df_final['Day_Cos'] = np.cos(2 * np.pi * df_final['Day_of_Week'] / 7)

# B. Shift Operasyonu (Gelecek Verisini Geçmişe Çevirme)
# App'te manuel girdiğimiz Rüzgar, Güneş vs. aslında Lag24 verisidir.
future_cols = ['Doğalgaz', 'Rüzgar', 'Güneş', 'Barajlı', 'Linyit', 'İthal Kömür', 'Akarsu', 'Fuel Oil', 'Jeotermal',
               'Biyokütle']
cols_to_shift = [c for c in future_cols if c in df_final.columns]
for col in cols_to_shift:
    df_final[f'{col}_Lag24'] = df_final[col].shift(24)
    # Orijinal sütunu silmiyoruz, analiz için kalsın ama modele sokmayacağız

# C. Fiyat Hafızası (Lags)
df_final['PTF_Lag_24'] = df_final[target_col].shift(24)
df_final['PTF_Lag_168'] = df_final[target_col].shift(168)

# D. İstatistiksel Özellikler (App.py bunları bekliyor!)
# ÖNEMLİ: App'te Roll_Std_24 = 50 olarak sabitlenmişti ama model burada doğrusunu öğrenmeli.
df_final['PTF_Roll_Mean_24'] = df_final['PTF_Lag_24'].rolling(24).mean()
df_final['PTF_Roll_Mean_168'] = df_final['PTF_Lag_24'].rolling(168).mean()
df_final['PTF_Roll_Std_24'] = df_final['PTF_Lag_24'].rolling(24).std()

# E. Sniper Özellikler
df_final['Relative_Price_Pos'] = (df_final['PTF_Lag_24'] - df_final['PTF_Roll_Mean_168']) / (
            df_final['PTF_Roll_Mean_168'] + 1)
df_final['Price_Momentum'] = df_final['PTF_Lag_24'] - df_final['PTF_Lag_168']

# F. Enerji Dengesi (Net Load & Thermal Stress)
ren_cols = ['Rüzgar_Lag24', 'Güneş_Lag24', 'Akarsu_Lag24', 'Jeotermal_Lag24', 'Biyokütle_Lag24']
existing_ren = [c for c in ren_cols if c in df_final.columns]
df_final['Total_Renewable_Lag24'] = df_final[existing_ren].sum(axis=1)

if load_col in df_final.columns:
    df_final['Net_Load'] = df_final[load_col] - df_final['Total_Renewable_Lag24']

    therm_cols = ['Doğalgaz_Lag24', 'İthal Kömür_Lag24', 'Linyit_Lag24', 'Fuel Oil_Lag24']
    existing_therm = [c for c in therm_cols if c in df_final.columns]
    df_final['Total_Thermal_Lag24'] = df_final[existing_therm].sum(axis=1)

    df_final['Thermal_Stress'] = df_final['Total_Thermal_Lag24'] / (df_final[load_col] + 1)
else:
    # Yük yoksa varsayılan
    df_final['Net_Load'] = 0
    df_final['Total_Thermal_Lag24'] = 0
    df_final['Thermal_Stress'] = 0

# Temizlik (NaN değerleri at)
df_final.dropna(inplace=True)

# -----------------------------------------------------------------------------
# 3. MODEL EĞİTİMİ (XGBOOST)
# -----------------------------------------------------------------------------
print("🔥 Model Eğitimi Başlıyor...")

# App.py'de input olarak hazırladığımız sütun listesiyle BURADAKİ aynı olmalı.
# Modele GİRMEYECEK sütunları atıyoruz.
exclude_cols = ['Tarih', 'Zaman', 'Saat', 'Saat_Int', 'Month', 'Day_of_Week', target_col]
# Ayrıca shift edilmemiş ham üretim sütunlarını da atalım (Data Leakage olmasın)
exclude_cols += cols_to_shift

feature_cols = [c for c in df_final.columns if c not in exclude_cols]
X = df_final[feature_cols]
y = df_final[target_col]

# Tarih Bazlı Bölme (Son 1 ayı teste ayır)
train_end_date = '2025-10-31'
train_mask = (df_final['Tarih'] <= train_end_date)
X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

# XGBoost Parametreleri (Daha önce optimize ettiklerimiz)
params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 5,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

print(f"✅ Model Eğitildi. Kullanılan Özellik Sayısı: {len(X_train.columns)}")
# print(f"Özellik Listesi: {X_train.columns.tolist()}") # Kontrol için açabilirsin

# -----------------------------------------------------------------------------
# 4. MODEL KAYDI
# -----------------------------------------------------------------------------
if not os.path.exists('models'):
    os.makedirs('models')

model_path = os.path.join('models', 'epias_model_final.pkl')

joblib.dump({
    'model': model,
    'features': X_train.columns.tolist(),  # App bu listeye bakarak input hazırlayacak
    'best_params': params
}, model_path)

print(f"📦 Model Paketi Hazır: {model_path}")
print("🏁 ŞİMDİ GÜVENLE 'streamlit run app.py' YAPABİLİRSİN KRAL!")